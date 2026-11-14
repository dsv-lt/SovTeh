"""Интеграционный тест MLOps-пайплайна в MLflow (Задание №1)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlflow.tracking import MlflowClient

from src.train_mlflow import REGISTRY_MODEL_NAME, run_experiments


class MLflowPipelineTests(unittest.TestCase):
    def test_mlflow_experiments_artifacts_and_staging_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "mlflow_test.db"
            tracking_uri = f"sqlite:///{db_path.as_posix()}"

            results, best, version = run_experiments(
                tracking_uri=tracking_uri,
                experiment_name="doc-classifier-test",
                model_name=f"{REGISTRY_MODEL_NAME}-test",
                grid=[(0.001, 2), (0.005, 2), (0.01, 2)],
                batch_size=32,
                force_synthetic=True,
            )

            self.assertEqual(len(results), 3)
            self.assertEqual(best.run_id, max(results, key=lambda r: (r.test_accuracy, r.test_f1_macro)).run_id)

            client = MlflowClient(tracking_uri=tracking_uri)
            exp = client.get_experiment_by_name("doc-classifier-test")
            self.assertIsNotNone(exp)

            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            self.assertEqual(len(runs), 3)
            for run in runs:
                self.assertIn("lr", run.data.params)
                self.assertIn("epochs", run.data.params)
                self.assertIn("test_accuracy", run.data.metrics)
                self.assertIn("test_f1_macro", run.data.metrics)
                self.assertEqual(run.data.tags.get("assignment"), "mlops-task-1")
                artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
                self.assertIn("config", artifacts)
                self.assertIn("charts", artifacts)

            mv = client.get_model_version(name=f"{REGISTRY_MODEL_NAME}-test", version=version)
            self.assertEqual(mv.tags.get("stage"), "Staging")
            if hasattr(mv, "aliases"):
                self.assertIn("staging", mv.aliases)


if __name__ == "__main__":
    unittest.main()
