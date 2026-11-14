"""Обучение PyTorch-классификатора стиля (TextClassifier) с полным MLOps-конвейером в MLflow.

Реализует Задание №1 (MLOps & AI-агенты):
1. Настройка хранилища ``sqlite:///mlflow.db`` и эксперимента ``doc-classifier``;
2. Функция ``train_once(lr, epochs, ...)`` с логированием параметров, метрик, тегов,
   дополнительных артефактов (график сравнения и JSON-конфиг) и модели
   через ``mlflow.pytorch.log_model`` (с сигнатурой и примером входа);
3. Перебор сетки гиперпараметров (по умолчанию ``lr in [0.0001, 0.001, 0.01]`` x
   ``epochs in [5, 10, 20]`` либо быстрый режим ``--quick`` из 4 запусков);
4. Регистрация лучшей модели в MLflow Model Registry и перевод в стадию ``Staging``.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.classifier import TextClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT / "data" / "processed" / "chunks_train.parquet"
MODELS_DIR = ROOT / "models"
DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "doc-classifier"
REGISTRY_MODEL_NAME = "literary-style-classifier"
SEED = 42

DEFAULT_GRID: list[tuple[float, int]] = [
    (lr, epochs)
    for lr in (0.0001, 0.001, 0.01)
    for epochs in (5, 10, 20)
]
QUICK_GRID: list[tuple[float, int]] = [
    (0.0001, 5),
    (0.0005, 5),
    (0.001, 5),
    (0.01, 5),
]


@dataclass
class DatasetSplit:
    """Разбиение признаков и меток на train/test с метаданными источника."""

    X_tr: sp.csr_matrix
    X_te: sp.csr_matrix
    y_tr: np.ndarray
    y_te: np.ndarray
    num_classes: int
    data_source: str


@dataclass
class RunResult:
    """Итоги одного MLflow-запуска."""

    run_id: str
    run_name: str
    lr: float
    epochs: int
    test_accuracy: float
    test_f1_macro: float
    model_uri: str = ""


def to_dense_batch(
    X_sparse: sp.csr_matrix, idx: np.ndarray | slice, device: torch.device
) -> torch.Tensor:
    """Конвертирует срез разреженной CSR-матрицы в плотный тензор на нужном устройстве."""
    return torch.from_numpy(X_sparse[idx].toarray()).float().to(device, non_blocking=True)


def _build_synthetic_split(in_features: int, num_classes: int, n_samples: int = 320) -> DatasetSplit:
    """Генерирует детерминированный разреженный датасет при отсутствии data/processed/.

    Использует реальную размерность TF-IDF (50 000) и 64 класса проекта, формируя
    классовые сигнатуры признаков для воспроизводимого прогона MLOps-конвейера.
    """
    rng = np.random.default_rng(SEED)
    active_classes = min(num_classes, 16)
    y = np.arange(n_samples) % active_classes

    # Назначаем каждому классу устойчивый набор характерных признаков + лёгкий шум
    class_anchors = [rng.choice(in_features, size=32, replace=False) for _ in range(active_classes)]
    rows, cols, vals = [], [], []
    for row_idx, cls_id in enumerate(y):
        chosen = rng.choice(class_anchors[cls_id], size=18, replace=False)
        noise = rng.choice(in_features, size=6, replace=False)
        feats = np.concatenate([chosen, noise])
        weights = np.concatenate([rng.uniform(0.3, 1.0, size=18), rng.uniform(0.05, 0.2, size=6)])
        rows.extend([row_idx] * len(feats))
        cols.extend(feats.tolist())
        vals.extend(weights.tolist())

    X = sp.csr_matrix((vals, (rows, cols)), shape=(n_samples, in_features), dtype=np.float32)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(splitter.split(X, y))
    return DatasetSplit(
        X_tr=X[tr_idx],
        X_te=X[te_idx],
        y_tr=y[tr_idx],
        y_te=y[te_idx],
        num_classes=num_classes,
        data_source="synthetic_fallback",
    )


def load_training_split(max_samples: int | None = None, force_synthetic: bool = False) -> DatasetSplit:
    """Загружает данные обучения из parquet или переключается на воспроизводимый fallback."""
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    num_classes = len(label_encoder.classes_)
    in_features = len(vectorizer.vocabulary_)

    if force_synthetic or not PROCESSED_PATH.exists():
        if not force_synthetic:
            logger.warning(
                "Файл %s не найден — используется воспроизводимый synthetic fallback "
                "(размерность %d признаков, %d классов).",
                PROCESSED_PATH,
                in_features,
                num_classes,
            )
        return _build_synthetic_split(in_features=in_features, num_classes=num_classes)

    train_df = pd.read_parquet(PROCESSED_PATH)
    if max_samples is not None and max_samples < len(train_df):
        train_df = train_df.sample(n=max_samples, random_state=SEED).reset_index(drop=True)

    X = vectorizer.transform(train_df["text"].tolist()).tocsr().astype(np.float32)
    y = label_encoder.transform(train_df["author"])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups=train_df["doc_id"]))
    return DatasetSplit(
        X_tr=X[tr_idx],
        X_te=X[te_idx],
        y_tr=y[tr_idx],
        y_te=y[te_idx],
        num_classes=num_classes,
        data_source="parquet",
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X_sparse: sp.csr_matrix,
    y_true: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[float, float]:
    """Оценивает accuracy и macro-F1 модели на тестовой выборке."""
    model.eval()
    logits_list: list[torch.Tensor] = []
    for start in range(0, X_sparse.shape[0], batch_size):
        end = min(start + batch_size, X_sparse.shape[0])
        xb = to_dense_batch(X_sparse, slice(start, end), device)
        logits_list.append(model(xb).cpu())
    logits = torch.cat(logits_list, dim=0)
    preds = logits.argmax(dim=1).numpy()
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    return acc, f1


def _save_run_artifacts(
    tmp_dir: Path,
    run_name: str,
    config_payload: dict[str, object],
    history: list[RunResult],
) -> tuple[Path, Path]:
    """Создаёт JSON-конфиг запуска и график сравнения точности для логирования как артефактов."""
    config_path = tmp_dir / "run_config.json"
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = [f"lr={r.lr}\nep={r.epochs}" for r in history]
    accs = [r.test_accuracy for r in history]
    f1s = [r.test_f1_macro for r in history]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    ax.bar(x - width / 2, accs, width, label="Test Accuracy", color="#2a9d8f")
    ax.bar(x + width / 2, f1s, width, label="Macro-F1", color="#e76f51")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title(f"MLflow Runs Comparison (latest: {run_name})")
    ax.legend()
    fig.tight_layout()

    chart_path = tmp_dir / "accuracy_chart.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    return config_path, chart_path


def _log_pytorch_model(model_cpu: nn.Module, sample_input: np.ndarray) -> str:
    """Логирует PyTorch-модель с сигнатурой и примером входа совместимо с MLflow 2.x/3.x."""
    with torch.no_grad():
        sample_output = model_cpu(torch.from_numpy(sample_input)).numpy()
    signature = infer_signature(sample_input, sample_output)
    pip_reqs = [f"torch=={torch.__version__.split('+')[0]}", f"numpy=={np.__version__}"]

    try:
        info = mlflow.pytorch.log_model(
            pytorch_model=model_cpu,
            name="model",
            signature=signature,
            input_example=sample_input,
            pip_requirements=pip_reqs,
        )
    except TypeError:
        info = mlflow.pytorch.log_model(
            pytorch_model=model_cpu,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
            pip_requirements=pip_reqs,
        )
    return str(info.model_uri)


def train_once(
    lr: float,
    epochs: int,
    split: DatasetSplit,
    device: torch.device,
    batch_size: int = 32,
    history: list[RunResult] | None = None,
) -> RunResult:
    """Один полный запуск обучения с логированием параметров, метрик, тегов, артефактов и модели."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = TensorDataset(
        torch.arange(split.X_tr.shape[0]),
        torch.as_tensor(split.y_tr, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    in_features = split.X_tr.shape[1]
    model = TextClassifier(in_features=in_features, num_classes=split.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    run_name = f"mlp_lr={lr}_ep={epochs}"
    with mlflow.start_run(run_name=run_name) as active_run:
        # 1. Параметры и теги эксперимента
        mlflow.log_params(
            {
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "in_features": in_features,
                "num_classes": split.num_classes,
                "optimizer": "Adam",
            }
        )
        mlflow.set_tags(
            {
                "assignment": "mlops-task-1",
                "model_family": "TextClassifier_MLP",
                "data_source": split.data_source,
            }
        )

        # 2. Цикл обучения
        t0 = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, total_samples = 0.0, 0
            for idx, yb in train_loader:
                xb = to_dense_batch(split.X_tr, idx.numpy(), device)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                total_samples += len(yb)
            mlflow.log_metric("train_loss", total_loss / max(1, total_samples), step=epoch)

        # 3. Валидация и итоговые метрики
        acc, f1_macro = evaluate(model, split.X_te, split.y_te, device=device)
        elapsed = time.perf_counter() - t0
        mlflow.log_metrics(
            {
                "test_accuracy": round(acc, 4),
                "test_f1_macro": round(f1_macro, 4),
                "train_time_sec": round(elapsed, 3),
            }
        )

        result = RunResult(
            run_id=active_run.info.run_id,
            run_name=run_name,
            lr=lr,
            epochs=epochs,
            test_accuracy=round(acc, 4),
            test_f1_macro=round(f1_macro, 4),
        )
        tracked_history = [*history, result] if history is not None else [result]

        # 4. Дополнительные артефакты (JSON-конфиг и график сравнения)
        with tempfile.TemporaryDirectory() as tmp:
            config_payload = {
                **asdict(result),
                "batch_size": batch_size,
                "in_features": in_features,
                "num_classes": split.num_classes,
                "data_source": split.data_source,
                "device": device.type,
            }
            config_path, chart_path = _save_run_artifacts(
                Path(tmp), run_name=run_name, config_payload=config_payload, history=tracked_history
            )
            mlflow.log_artifact(str(config_path), artifact_path="config")
            mlflow.log_artifact(str(chart_path), artifact_path="charts")

        # 5. Артефакт-модель PyTorch
        model_cpu = model.to("cpu").eval()
        sample_input = split.X_te[:2].toarray().astype(np.float32)
        result.model_uri = _log_pytorch_model(model_cpu, sample_input)

        logger.info(
            "[MLflow] %s | test_accuracy=%.4f | macro-F1=%.4f | run_id=%s",
            run_name,
            acc,
            f1_macro,
            active_run.info.run_id,
        )
        return result


def register_best_model_to_staging(
    results: list[RunResult],
    model_name: str = REGISTRY_MODEL_NAME,
) -> tuple[RunResult, str]:
    """Регистрирует лучшую модель в MLflow Model Registry и переводит в Staging."""
    if not results:
        raise ValueError("Список результатов запусков пуст — нечего регистрировать")

    best = max(results, key=lambda r: (r.test_accuracy, r.test_f1_macro))
    model_uri = best.model_uri or f"runs:/{best.run_id}/model"
    reg = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(reg.version)

    client = MlflowClient()
    if hasattr(client, "set_registered_model_alias"):
        client.set_registered_model_alias(model_name, "staging", version)
    client.set_model_version_tag(model_name, version, "stage", "Staging")

    if hasattr(client, "transition_model_version_stage"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Staging",
                )
            except MlflowException as exc:
                logger.warning("Переход стадии через deprecated API пропущен: %s", exc)

    logger.info(
        "Лучшая модель '%s' v%s (run=%s, acc=%.4f, macro-F1=%.4f) переведена в Staging.",
        model_name,
        version,
        best.run_name,
        best.test_accuracy,
        best.test_f1_macro,
    )
    return best, version


def run_experiments(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    model_name: str = REGISTRY_MODEL_NAME,
    grid: list[tuple[float, int]] | None = None,
    batch_size: int = 32,
    max_samples: int | None = None,
    force_synthetic: bool = False,
) -> tuple[list[RunResult], RunResult, str]:
    """Полный MLOps-конвейер: подготовка данных, серия запусков и регистрация в Staging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("MLflow tracking_uri=%s | experiment=%s | device=%s", tracking_uri, experiment_name, device)

    split = load_training_split(max_samples=max_samples, force_synthetic=force_synthetic)
    run_grid = grid or (QUICK_GRID if split.data_source == "synthetic_fallback" else DEFAULT_GRID)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    results: list[RunResult] = []
    for lr, epochs in run_grid:
        res = train_once(
            lr=lr,
            epochs=epochs,
            split=split,
            device=device,
            batch_size=batch_size,
            history=results,
        )
        results.append(res)

    best, version = register_best_model_to_staging(results, model_name=model_name)
    return results, best, version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLflow эксперименты для TextClassifier (Задание №1)")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI, help="URI хранилища MLflow")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="Имя эксперимента MLflow")
    parser.add_argument("--model-name", default=REGISTRY_MODEL_NAME, help="Имя модели в Model Registry")
    parser.add_argument("--quick", action="store_true", help="Быстрая сетка из 4 запусков")
    parser.add_argument("--synthetic", action="store_true", help="Использовать synthetic fallback вместо parquet")
    parser.add_argument("--max-samples", type=int, default=None, help="Ограничение числа примеров из parquet")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    grid = QUICK_GRID if args.quick else None
    run_experiments(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        model_name=args.model_name,
        grid=grid,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        force_synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
