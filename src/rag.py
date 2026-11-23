"""Семантическое ядро RAG (Локальный ML-фильтр + RuBERT / FAISS векторный поиск по сюжетам).

Двухэтапный конвейер обработки запроса:
1. Локальный ML-фильтр (`HashingVectorizer` + `SGDClassifier` на 7 макро-жанров)
   совместно с нейросетевой атрибуцией стиля (`StyleClassifier` на 64 автора)
   моментально определяет жанровую и авторскую направленность вопроса пользователя.
2. Семантический поиск (RuBERT-эмбеддинги с поддержкой CUDA GPU / гибридных плотных
   векторов + индекс ``faiss.IndexFlatIP`` в оперативной памяти или PyTorch/NumPy fallback)
   извлекает релевантные описания сюжетов из подготовленного датасета ``data/plots_dataset.csv``.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from operator import itemgetter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from src.config import PROJECT_ROOT, Settings, get_settings
from src.llm_client import LLMClient
from src.preprocessing import (
    configure_cross_platform_stdio,
    enforce_network_immunity,
    read_text_cross_platform,
)

enforce_network_immunity()
configure_cross_platform_stdio()

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import-not-found]
except ImportError:
    faiss = None

STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "найди", "фильм", "книга", "книгу", "произведение", "расскажи", "сюжет",
    "про", "какой", "какие", "который", "очень", "кино", "или", "похожий",
    "вроде", "типа", "посоветуй", "есть", "чо", "литературный", "текст",
}

RU_ENDINGS = re.compile(
    r"(ами|ями|ого|его|ому|ему|ыми|ими|овать|евать|ться|тись|ать|ять|ить|еть|уть|ыть|"
    r"ов|ев|ей|ам|ям|ах|ях|ом|ем|ой|ий|ый|ая|яя|ое|ее|ые|ие|у|ю|а|я|о|е|ы|и|й|ь)$"
)

_QUERY_STOP_ROOTS = {
    "найд", "фильм", "книг", "расскаж", "сюжет", "про", "как", "котор",
    "очен", "кин", "ил", "похож", "врод", "тип", "посовету", "произведен",
    "истор", "автор", "покаж", "напиш",
}

knowledge_base: list[dict[str, Any]] = []
ml_vectorizer: HashingVectorizer | None = None
ml_classifier: SGDClassifier | None = None
vector_index: FAISSVectorIndex | None = None


def clean_wiki_garbage(text: str) -> str:
    """Очищает HTML/Wiki-артефакты и лишние пробелы из описаний."""
    if not text:
        return ""
    text = re.sub(r"mwparseroutput.*?(?=\s[А-ЯЁA-Z]|$)", "", text)
    text = re.sub(r"[a-zA-Z0-9_-]{24,}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def collapse_genres(raw_genre: str) -> str:
    """Преобразует сырые жанровые теги в 7 базовых макро-категорий."""
    if not raw_genre:
        return "драма и мелодрама"
    raw = str(raw_genre).lower()

    if any(w in raw for w in ("фантаст", "фэнтез", "космос", "марс", "киберпанк", "sci-fi", "fantasy")):
        return "фантастика и фэнтези"
    if any(w in raw for w in ("ужас", "хоррор", "мистик", "призрак", "готич", "horror")):
        return "ужасы и мистика"
    if any(w in raw for w in ("комед", "сатир", "юмор", "плутов", "comedy")):
        return "комедия и сатира"
    if any(w in raw for w in ("военн", "война", "арми", "сражен", "истори", "смут", "war")):
        return "история и война"
    if any(w in raw for w in ("боевик", "экшен", "детектив", "кримин", "преступ", "сыск", "подполь")):
        return "боевик и детектив"
    if "триллер" in raw:
        return "ужасы и мистика"
    if any(w in raw for w in ("документаль", "хроника", "мемуар", "письм", "очерк", "биограф")):
        return "документальный и хроника"
    if any(w in raw for w in ("драма", "мелодрама", "романти", "трагед", "drama")):
        return "драма и мелодрама"
    return "драма и мелодрама"


def clean_and_normalize_text(text: str) -> str:
    """Очищает спецсимволы и выполняет лёгкий стемминг русских слов."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"[^а-яёa-z0-9\s]", " ", text.lower())
    tokens: list[str] = []
    for word in cleaned.split():
        if word not in STOP_WORDS and len(word) > 2:
            if len(word) > 3:
                word = RU_ENDINGS.sub("", word)
            if len(word) >= 3:
                tokens.append(word)
    return " ".join(tokens)


def stem_russian_text(text: str) -> str:
    """Алиас морфологической нормализации для совместимости."""
    return clean_and_normalize_text(text)


def _resolve_rubert_encoder(classifier: Any | None, device: torch.device) -> tuple[Any, Any] | None:
    """Извлекает уже загруженный в память RuBERT-энкодер (без сетевых обращений)."""
    if classifier is None:
        return None
    tokenizer = getattr(classifier, "tokenizer", None)
    model = getattr(classifier, "model", None)
    if tokenizer is None or model is None:
        return None
    backbone = getattr(model, "bert", None) or getattr(model, "base_model", None)
    if backbone is None:
        return None
    try:
        backbone.to(device).eval()
    except (RuntimeError, ValueError):
        return None
    return tokenizer, backbone


class FAISSVectorIndex:
    """Векторный индекс в памяти (RuBERT/TF-IDF + ``faiss.IndexFlatIP`` или PyTorch/NumPy)."""

    def __init__(
        self,
        texts: list[str],
        device: str = "auto",
        classifier: Any | None = None,
        prefer_rubert: bool = False,
    ) -> None:
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        )
        self.backend_name = "tfidf-cosine"
        self._tfidf: TfidfVectorizer | None = None
        self._tokenizer: Any = None
        self._backbone: Any = None
        self._faiss_index: Any = None
        self._matrix: np.ndarray = np.zeros((0, 1), dtype=np.float32)

        if not texts:
            return

        encoder = _resolve_rubert_encoder(classifier, self.device) if prefer_rubert else None
        if encoder is not None:
            self._tokenizer, self._backbone = encoder
            self.backend_name = f"rubert-{self.device.type}"
            embeddings = self._encode_rubert(texts)
        else:
            self._tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=4096,
                sublinear_tf=True,
            )
            sparse_mat = self._tfidf.fit_transform(texts)
            embeddings = sparse_mat.toarray().astype(np.float32)
            self.backend_name = "faiss-hybrid-dense" if faiss is not None else "pytorch-hybrid-dense"

        self._matrix = self._l2_normalize(embeddings)
        if faiss is not None and self._matrix.shape[0] > 0:
            idx = faiss.IndexFlatIP(self._matrix.shape[1])
            idx.add(self._matrix)
            self._faiss_index = idx
            if "faiss" not in self.backend_name:
                self.backend_name = f"{self.backend_name}+faiss"

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        if mat.size == 0:
            return mat.astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (mat / norms).astype(np.float32)

    def _encode_rubert(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        assert self._tokenizer is not None and self._backbone is not None
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self._tokenizer(
                    batch,
                    truncation=True,
                    max_length=256,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self._backbone(**enc)
                hidden = outputs.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                chunks.append(pooled.float().cpu().numpy())
        return np.vstack(chunks).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if self._backbone is not None:
            vec = self._encode_rubert([query])
        elif self._tfidf is not None:
            vec = self._tfidf.transform([query]).toarray().astype(np.float32)
        else:
            return np.zeros((1, 1), dtype=np.float32)
        return self._l2_normalize(vec)

    def search_scores(self, query: str) -> np.ndarray:
        """Возвращает массив косинусных сходств длиной N по всем сюжетам индекса."""
        if self._matrix.shape[0] == 0:
            return np.array([], dtype=np.float32)
        q_vec = self.encode_query(query)
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_vec, self._matrix.shape[0])
            ordered = np.zeros(self._matrix.shape[0], dtype=np.float32)
            for sc, idx in zip(scores[0], indices[0], strict=False):
                if idx >= 0:
                    ordered[int(idx)] = float(sc)
            return ordered
        if self.device.type == "cuda":
            q_t = torch.from_numpy(q_vec).to(self.device)
            m_t = torch.from_numpy(self._matrix).to(self.device)
            return torch.matmul(m_t, q_t.T).squeeze(-1).cpu().numpy()
        return (self._matrix @ q_vec.T).squeeze(-1)


def resolve_dataset_path(dataset_path: Path | str | None = None) -> Path | None:
    """Находит подготовленный CSV-датасет сюжетов (data/plots_dataset.csv или films_data.csv)."""
    candidates: list[Path] = []
    if dataset_path is not None:
        candidates.append(Path(dataset_path))
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "plots_dataset.csv",
            PROJECT_ROOT / "plots_dataset.csv",
            PROJECT_ROOT / "films_data.csv",
        ]
    )
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def load_plots_dataset(dataset_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Загружает датасет сюжетов с поддержкой кодировок UTF-8 и CP1251."""
    resolved = resolve_dataset_path(dataset_path)
    if resolved is None:
        logger.warning("Подготовленный датасет сюжетов не найден.")
        return []

    raw_text = read_text_cross_platform(resolved)
    records: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(raw_text))
    for idx, row in enumerate(reader, start=1):
        title = (row.get("title") or row.get("name") or f"Произведение #{idx}").strip()
        author = (row.get("author") or "Неизвестный автор").strip()
        raw_genre = clean_wiki_garbage(row.get("genre") or "")
        year = (row.get("year") or "").strip()
        rating_raw = str(row.get("rating") or row.get("imdb_rating") or "0").replace(",", ".").strip()
        try:
            rating_num = float(rating_raw)
        except ValueError:
            rating_num = 0.0

        summary = clean_wiki_garbage(row.get("summary") or "")
        plot = clean_wiki_garbage(row.get("plot") or row.get("description") or "")
        full_plot = f"{summary} {plot}".strip()
        if not full_plot:
            continue

        macro_genre = collapse_genres(raw_genre)
        rating_label = f"{rating_num:.1f}" if rating_num > 0 else "Н/Д"
        year_part = f", {year} г." if year else ""
        display = (
            f"Произведение: «{title}» | Автор: {author}{year_part} | "
            f"Жанр: {raw_genre or macro_genre} (Рейтинг: {rating_label})\n"
            f"Сюжет: {full_plot[:850]}"
        )
        searchable = f"{title} {author} {raw_genre} {macro_genre} {full_plot}"
        records.append(
            {
                "id": row.get("id") or str(idx),
                "title": title,
                "author": author,
                "genre": macro_genre,
                "raw_genre": raw_genre or macro_genre,
                "year": year,
                "rating": rating_num,
                "summary": summary,
                "plot": plot,
                "stemmed_search": clean_and_normalize_text(searchable),
                "lower_text": display.lower(),
                "display_text": display,
            }
        )
    return records


def train_or_load_genre_filter(
    records: list[dict[str, Any]],
    model_path: Path | None = None,
    vectorizer_path: Path | None = None,
) -> tuple[HashingVectorizer | None, SGDClassifier | None]:
    """Загружает или обучает быстрый локальный ML-фильтр жанров (HashingVectorizer + SGDClassifier)."""
    if (
        vectorizer_path is not None
        and model_path is not None
        and vectorizer_path.exists()
        and model_path.exists()
    ):
        try:
            vec = joblib.load(vectorizer_path)
            clf = joblib.load(model_path)
            if hasattr(vec, "transform") and hasattr(clf, "predict_proba"):
                return vec, clf
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось загрузить сохранённый ML-фильтр (%s), переобучаем в памяти.", exc)

    if not records:
        return None, None

    texts: list[str] = []
    labels: list[str] = []
    for item in records:
        stemmed = item.get("stemmed_search", "")
        if stemmed:
            texts.append(stemmed)
            labels.append(str(item["genre"]))

    unique_labels = set(labels)
    if len(texts) < 4 or len(unique_labels) < 2:
        return None, None

    vec = HashingVectorizer(ngram_range=(1, 2), n_features=2**16, alternate_sign=False)
    clf = SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42, max_iter=1000)
    X = vec.transform(texts)
    clf.fit(X, labels)
    return vec, clf


def predict_genre(question: str) -> tuple[str, list[str], float]:
    """Определяет вероятный жанр запроса с помощью локального ML-фильтра."""
    cleaned = clean_and_normalize_text(question) or question.lower()
    if ml_vectorizer is not None and ml_classifier is not None and cleaned.strip():
        vec = ml_vectorizer.transform([cleaned])
        if hasattr(ml_classifier, "predict_proba"):
            probs = ml_classifier.predict_proba(vec)[0]
            top_idx = np.argsort(-probs)[:2]
            top_genres = [str(ml_classifier.classes_[i]) for i in top_idx]
            return top_genres[0], top_genres, float(probs[top_idx[0]])
        pred = str(ml_classifier.predict(vec)[0])
        return pred, [pred], 1.0

    # Правило-эвристический fallback, если фильтр не инициализирован
    inferred = collapse_genres(question)
    return inferred, [inferred], 0.5


def build_index(
    settings: Settings | None = None,
    classifier: Any | None = None,
    dataset_path: Path | str | None = None,
) -> int:
    """Инициализирует базу сюжетов, локальный ML-фильтр жанров и векторный индекс RuBERT/FAISS."""
    global ml_vectorizer, ml_classifier, vector_index

    cfg = settings or get_settings()
    records = load_plots_dataset(dataset_path or cfg.dataset_path)
    knowledge_base.clear()
    knowledge_base.extend(records)

    ml_vectorizer, ml_classifier = train_or_load_genre_filter(
        knowledge_base,
        model_path=cfg.genre_model_path,
        vectorizer_path=cfg.hashing_vectorizer_path,
    )

    prefer_rubert = cfg.rag_embedding_backend == "rubert" or (
        cfg.rag_embedding_backend == "auto" and classifier is not None and hasattr(classifier, "tokenizer")
    )
    corpus_texts = [item["display_text"] for item in knowledge_base]
    vector_index = FAISSVectorIndex(
        texts=corpus_texts,
        device=cfg.device,
        classifier=classifier,
        prefer_rubert=prefer_rubert,
    )

    logger.info(
        "RAG-индекс собран: %d сюжетов | ML-фильтр=%s | векторный бэкенд=%s",
        len(knowledge_base),
        "Hashing+SGD" if ml_classifier is not None else "fallback",
        vector_index.backend_name if vector_index else "none",
    )
    return len(knowledge_base)


def _extract_query_roots(question: str) -> list[str]:
    raw_words = re.sub(r"[^а-яёa-z0-9\s]", " ", question.lower()).split()
    roots: list[str] = []
    for word in raw_words:
        if len(word) < 3 or word in STOP_WORDS:
            continue
        root = RU_ENDINGS.sub("", word) if len(word) > 4 else word
        if len(root) >= 3 and root not in _QUERY_STOP_ROOTS and root not in roots:
            roots.append(root)
    return roots


def rag_search(
    question: str,
    k: int = 3,
    predicted_author: str | None = None,
) -> tuple[str, list[str], list[dict[str, Any]], bool]:
    """Двухэтапный гибридный поиск: ML-фильтр жанра/автора + RuBERT/FAISS векторное ранжирование."""
    if not knowledge_base:
        build_index()

    top_genre, top_genres, _ = predict_genre(question)
    if not knowledge_base:
        return top_genre, [], [], False

    vec_scores = (
        vector_index.search_scores(question)
        if vector_index is not None
        else np.zeros(len(knowledge_base), dtype=np.float32)
    )
    user_roots = _extract_query_roots(question)
    q_lower = question.lower()
    author_surname = predicted_author.split()[0].lower() if predicted_author else ""

    scored: list[dict[str, Any]] = []
    for idx, item in enumerate(knowledge_base):
        text_lower = item["lower_text"]
        stemmed_text = item["stemmed_search"]

        # 1. Лексическое совпадение по корням и подстрокам
        root_hits = 0
        freq_hits = 0
        for root in user_roots:
            if root in text_lower or root in stemmed_text:
                root_hits += 1
                freq_hits += text_lower.count(root) + stemmed_text.count(root)

        # 2. Семантическое сходство RuBERT/FAISS
        cos_sim = float(vec_scores[idx]) if idx < len(vec_scores) else 0.0

        # 3. Бонус локального ML-фильтра (жанр + авторский стиль)
        genre_bonus = 0.12 if item["genre"] in top_genres else 0.0
        author_bonus = 0.15 if (author_surname and author_surname in item["author"].lower()) else 0.0
        title_bonus = 2.0 if item["title"].lower() in q_lower else 0.0

        total_score = (root_hits * 1.5) + (freq_hits * 0.15) + (cos_sim * 2.5) + title_bonus
        if root_hits > 0 or cos_sim >= 0.18:
            total_score += genre_bonus + author_bonus

        scored.append(
            {
                "score": total_score,
                "root_hits": root_hits,
                "cos_sim": cos_sim,
                "item": item,
            }
        )

    scored.sort(key=lambda x: (x["score"], x["item"]["rating"]), reverse=True)
    best = scored[0] if scored else None
    exact_match = bool(best and (best["root_hits"] > 0 or best["cos_sim"] >= 0.18))

    if exact_match:
        chosen = [entry["item"] for entry in scored[: max(1, k)]]
    else:
        rated_fallback = sorted(knowledge_base, key=itemgetter("rating"), reverse=True)
        chosen = rated_fallback[: max(1, k)]

    docs = [item["display_text"] for item in chosen]
    return top_genre, docs, chosen, exact_match


async def rag_answer_async(
    question: str,
    llm_client: LLMClient,
    k: int = 3,
    predicted_author: str = "не определён",
    temperature: float = 0.2,
) -> tuple[str, list[str], str]:
    """Асинхронный полный RAG-пайплайн: поиск по сюжетной базе + генерация ответа через LLM API."""
    top_genre, docs, _, exact_match = rag_search(
        question=question,
        k=k,
        predicted_author=predicted_author,
    )
    if not docs:
        return "Локальная база сюжетов пуста. Проверьте наличие data/plots_dataset.csv.", [], top_genre

    context_text = "\n\n".join(docs)
    answer = await llm_client.generate_with_context(
        question=question,
        context_text=context_text,
        predicted_author=predicted_author,
        predicted_genre=top_genre,
        exact_match=exact_match,
        temperature=temperature,
    )
    return answer, docs, top_genre


async def run_analyst_agent_async(
    question: str,
    llm_client: LLMClient,
    temperature: float = 0.1,
) -> tuple[str, list[str]]:
    """Агент-Аналитик: вычисляет реальную статистику рейтингов и жанров по базе сюжетов."""
    if not knowledge_base:
        build_index()

    steps: list[str] = ["🔍 Шаг 1: Сканирование локальной базы сюжетов для расчёта статистики..."]
    rated = [item for item in knowledge_base if item["rating"] > 0]
    avg_total = round(sum(x["rating"] for x in rated) / len(rated), 2) if rated else 0.0
    steps.append(
        f"📊 Шаг 2: Всего произведений: {len(knowledge_base)} (с рейтингом: {len(rated)}, средний рейтинг: {avg_total})."
    )

    roots = _extract_query_roots(question)
    matched = [
        item
        for item in rated
        if any(r in item["lower_text"] or r in item["stemmed_search"] for r in roots)
    ]

    if matched:
        avg_sub = round(sum(x["rating"] for x in matched) / len(matched), 2)
        best = max(matched, key=itemgetter("rating"))
        worst = min(matched, key=itemgetter("rating"))
        steps.append(
            f"🧮 Шаг 3: По выборке ({len(matched)} произв.) средний рейтинг = {avg_sub}."
        )
        stats_context = (
            f"Всего произведений в базе: {len(knowledge_base)} (средний рейтинг: {avg_total}).\n"
            f"Найдено по выборке: {len(matched)}, средний рейтинг выборки: {avg_sub}.\n"
            f"Лучшее произведение выборки:\n{best['display_text']}\n\n"
            f"Произведение с наименьшим рейтингом в выборке:\n{worst['display_text']}"
        )
    else:
        steps.append("⚠️ Шаг 3: Узкая выборка пуста, приведена общая сводка по базе.")
        genre_stats: dict[str, list[float]] = {}
        for item in rated:
            genre_stats.setdefault(item["genre"], []).append(item["rating"])
        summary_lines = [
            f"- {g}: {len(vals)} произв., ср. рейтинг {round(sum(vals) / len(vals), 2)}"
            for g, vals in sorted(genre_stats.items())
        ]
        stats_context = (
            f"Всего произведений в базе: {len(knowledge_base)}, средний рейтинг: {avg_total}.\n"
            "Сводка по макро-жанрам:\n" + "\n".join(summary_lines)
        )

    sys_prompt = (
        "Ты — Агент-Аналитик базы литературных сюжетов. Отвечай строго с опорой "
        "на переданные расчёты, цифры и выписку. Не выдумывай несуществующих значений."
    )
    user_msg = f"Результаты расчёта по базе:\n{stats_context}\n\nВопрос пользователя: {question}"
    answer = await llm_client.generate(
        system_prompt=sys_prompt,
        user_message=user_msg,
        temperature=temperature,
    )
    return answer, steps


async def run_critic_agent_async(
    question: str,
    llm_client: LLMClient,
    temperature: float = 0.2,
) -> tuple[str, list[str]]:
    """Агент-Критик: подбирает до 4 кандидатов и сравнивает их по атмосфере, сюжету и стилю."""
    if not knowledge_base:
        build_index()

    steps: list[str] = ["🎭 Шаг 1: Семантический подбор кандидатов из базы сюжетов..."]
    top_genre, docs, _, exact_match = rag_search(question=question, k=4)

    if exact_match:
        steps.append(
            f"⚖️ Шаг 2: Отобрано {len(docs)} релевантных произведений (жанр: {top_genre}) для сравнения."
        )
        prefix = ""
    else:
        steps.append("⚠️ Шаг 2: Точных совпадений не найдено, отобраны топ-произведения по рейтингу.")
        prefix = (
            "В базе нет точных совпадений по вашему описанию, поэтому как литературный критик "
            "я сравниваю эти высокооценённые произведения из датасета:\n"
        )

    sys_prompt = (
        "Ты — Агент-Критик и литературный эксперт. Сравни предложенные произведения из выписки "
        "по атмосфере, конфликту сюжета, авторскому стилю и рейтингу, и дай аргументированную "
        "рекомендацию под запрос пользователя. Запрещено выдумывать факты вне выписки."
    )
    user_msg = f"{prefix}Кандидаты из базы:\n" + "\n\n".join(docs) + f"\n\nЗапрос пользователя: {question}"
    answer = await llm_client.generate(
        system_prompt=sys_prompt,
        user_message=user_msg,
        temperature=temperature,
    )
    return answer, steps + docs
