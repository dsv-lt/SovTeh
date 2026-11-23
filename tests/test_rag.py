"""Тесты гибридной RAG-системы, ML-фильтра, векторного поиска и CP1251/UTF-8 кодирования."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from src.config import PROJECT_ROOT, Settings
from src.preprocessing import read_text_cross_platform
from src.rag import (
    FAISSVectorIndex,
    build_index,
    clean_and_normalize_text,
    collapse_genres,
    load_plots_dataset,
    predict_genre,
    rag_search,
    run_analyst_agent_async,
    run_critic_agent_async,
)


class RAGPipelineTests(unittest.TestCase):
    def test_dataset_covers_all_64_authors_and_cross_platform_encodings(self) -> None:
        records = load_plots_dataset(PROJECT_ROOT / "data" / "plots_dataset.csv")
        self.assertGreaterEqual(len(records), 128)

        labels = json.loads(
            (PROJECT_ROOT / "models" / "lora_classifier" / "labels.json").read_text(encoding="utf-8")
        )["classes"]
        dataset_authors = {item["author"] for item in records}
        self.assertTrue(set(labels).issubset(dataset_authors))

        with tempfile.TemporaryDirectory() as tmp:
            p_cp1251 = Path(tmp) / "cp1251_sample.txt"
            p_utf8_bom = Path(tmp) / "utf8_bom_sample.txt"
            sample = "Космическая экспедиция к Марсу и тайны Петербурга."
            p_cp1251.write_bytes(sample.encode("cp1251"))
            p_utf8_bom.write_bytes(b"\xef\xbb\xbf" + sample.encode("utf-8"))
            self.assertEqual(read_text_cross_platform(p_cp1251), sample)
            self.assertEqual(read_text_cross_platform(p_utf8_bom), sample)

    def test_genre_collapse_stemming_and_ml_filter(self) -> None:
        self.assertEqual(collapse_genres("Космос и научная фантастика"), "фантастика и фэнтези")
        self.assertEqual(collapse_genres("Военно-исторический роман"), "история и война")
        self.assertEqual(collapse_genres("Детектив и триллер"), "боевик и детектив")
        self.assertEqual(collapse_genres("Мистика и ужасы"), "ужасы и мистика")

        stemmed = clean_and_normalize_text("Космическая экспедиция к Красной планете Марс")
        self.assertIn("экспедиц", stemmed)
        self.assertIn("марс", stemmed)

        count = build_index(Settings(classifier_backend="baseline", _env_file=None))
        self.assertGreaterEqual(count, 128)
        top_genre, top_2, conf = predict_genre("космический корабль летит на Марс к пришельцам")
        self.assertIn(top_genre, top_2)
        self.assertGreater(conf, 0.0)

    def test_semantic_vector_index_and_rag_search_exact_and_fallback(self) -> None:
        idx = FAISSVectorIndex(
            [
                "Инженер Лось отправляется на космическом корабле к планете Марс.",
                "Петербургский студент Раскольников совершает преступление в июле.",
                "Уездный доктор скучает в старой усадьбе за вишнёвым садом.",
            ]
        )
        scores = idx.search_scores("полёт на Марс на космическом корабле")
        self.assertEqual(len(scores), 3)
        self.assertEqual(int(scores.argmax()), 0)

        build_index(Settings(classifier_backend="baseline", _env_file=None))
        genre, docs, items, exact = rag_search(
            "Найди произведение про космическую экспедицию на Марс",
            k=2,
            predicted_author="Толстой А",
        )
        self.assertTrue(exact)
        self.assertEqual(genre, "фантастика и фэнтези")
        self.assertTrue(any("Аэлита" in d for d in docs))
        self.assertEqual(items[0]["title"], "Аэлита")

        # Запрос без совпадений в базе -> честный fallback на высокий рейтинг
        _, fb_docs, fb_items, fb_exact = rag_search("qwertyuiopasdfghjklzxcvbnm12345", k=3)
        self.assertFalse(fb_exact)
        self.assertEqual(len(fb_docs), 3)
        ratings = [it["rating"] for it in fb_items]
        self.assertEqual(ratings, sorted(ratings, reverse=True))


class RAGMultiAgentModesTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyst_and_critic_agents(self) -> None:
        build_index(Settings(classifier_backend="baseline", _env_file=None))

        class StubLLM:
            def __init__(self) -> None:
                self.generate = AsyncMock(return_value="Сформированный ответ агента.")

        stub = StubLLM()
        ans_a, steps_a = await run_analyst_agent_async("Средний рейтинг фантастики", stub)
        self.assertEqual(ans_a, "Сформированный ответ агента.")
        self.assertGreaterEqual(len(steps_a), 3)

        ans_c, steps_c = await run_critic_agent_async("Что почитать про мистический Петербург?", stub)
        self.assertEqual(ans_c, "Сформированный ответ агента.")
        self.assertGreaterEqual(len(steps_c), 3)


if __name__ == "__main__":
    unittest.main()
