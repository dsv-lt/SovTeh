"""Юнит- и интеграционные тесты инструментов и трёх архитектур ИИ-агентов (Задание №2)."""

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import torch
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents_comparison import MockCompositeChatModel
from src.agents import (
    TOOL_SCHEMAS,
    TOOLS,
    extract_json,
    run_framework_agent,
    run_json_agent,
    run_native_agent,
    run_tool,
    set_agent_classifier,
    tool_author_metrics,
    tool_calc,
    tool_classify_style,
    tool_search_plots,
)
from src.app import app
from src.classifier import StyleClassifier
from src.config import Settings
from src.llm_client import LLMChatResult, LLMClient, LLMError, ToolCall


class DummyAgentClassifier(StyleClassifier):
    name = "dummy-agent-classifier"

    def __init__(self) -> None:
        super().__init__(Settings(llm_api_key="gsk_dummy", _env_file=None))
        self.classes = ["Чехов А", "Достоевский Ф", "Пушкин А"]

    def _logits(self, chunks: list[str]) -> torch.Tensor:
        return torch.tensor([[3.5, 0.8, -0.5] for _ in chunks], dtype=torch.float32)


class LoopingChatModel(BaseChatModel):
    """Всегда запрашивает tool_call, чтобы проверить GraphRecursionError / max_iterations."""

    @property
    def _llm_type(self) -> str:
        return "looping-chat"

    def bind_tools(self, tools: Any, **_kwargs: Any) -> LoopingChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = AIMessage(
            content="",
            tool_calls=[{"id": f"loop_{len(messages)}", "name": "calc", "args": {"expression": "1+1"}}],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


class AgentToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        set_agent_classifier(DummyAgentClassifier())

    def tearDown(self) -> None:
        set_agent_classifier(None)

    def test_tool_calc_valid_and_error_cases(self) -> None:
        self.assertEqual(tool_calc("2+2"), "4")
        self.assertEqual(tool_calc("2*(3+4)"), "14")
        self.assertEqual(tool_calc("0.9394 - 0.8934"), "0.046")
        self.assertIn("Ошибка", tool_calc("2+"))
        self.assertIn("Ошибка", tool_calc("1/0"))
        self.assertIn("Ошибка", tool_calc("__import__('os')"))

    def test_tool_author_metrics_and_classify_style(self) -> None:
        dost = json.loads(tool_author_metrics("Достоевский"))
        self.assertEqual(dost["author"], "Достоевский Ф")
        self.assertGreater(dost["f1_score"], 0.9)

        overall = json.loads(tool_author_metrics("overall"))
        self.assertIn("lora", overall)

        self.assertIn("не найден", tool_author_metrics("Несуществующий Автор 123"))

        clf_res = json.loads(tool_classify_style("В саду за усадьбой тихо осыпались липы."))
        self.assertEqual(clf_res["predicted_author"], "Чехов А")
        self.assertEqual(len(clf_res["top_3"]), 3)

        plots_res = json.loads(tool_search_plots("космическая экспедиция к Марсу"))
        self.assertTrue(plots_res["exact_match"])
        self.assertTrue(any("Аэлита" in item for item in plots_res["results"]))

    def test_run_tool_handles_unknown_and_validation_errors(self) -> None:
        err_unknown = run_tool("lookup", {"query": "Чехов"})
        self.assertIn("неизвестный инструмент", err_unknown)
        self.assertIn("classify_style", err_unknown)

        err_extra = run_tool("calc", {"expression": "2+2", "unexpected": 1})
        self.assertIn("Ошибка валидации", err_extra)

        err_empty = run_tool("classify_style", "")
        self.assertIn("Ошибка", err_empty)

        self.assertEqual(run_tool("calc", "2+3"), "5")
        self.assertEqual(len(TOOL_SCHEMAS), len(TOOLS))
        self.assertEqual({s["function"]["name"] for s in TOOL_SCHEMAS}, set(TOOLS.keys()))

    def test_extract_json_handles_wrappers(self) -> None:
        self.assertEqual(extract_json('Вот ответ: {"action": "calc", "input": "2+2"}!'), {"action": "calc", "input": "2+2"})
        self.assertEqual(extract_json('```json\n{"action": "final", "answer": "4"}\n```'), {"action": "final", "answer": "4"})
        with self.assertRaises(ValueError):
            extract_json("Просто текст без JSON")


class AgentLoopsMockTests(unittest.TestCase):
    def setUp(self) -> None:
        set_agent_classifier(DummyAgentClassifier())

    def tearDown(self) -> None:
        set_agent_classifier(None)

    def test_json_agent_multistep_recovery_and_max_iterations(self) -> None:
        calls = 0

        def mock_llm(prompt: str) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                return "Это не JSON вообще"
            if calls == 2:
                return 'Вот: {"action": "lookup", "input": "Чехов"}'
            if calls == 3:
                return '```json\n{"action": "author_metrics", "input": "Чехов А"}\n```'
            if calls == 4:
                return '{"action": "calc", "input": "0.9394 - 0.8934"}'
            return '{"action": "final", "answer": "Разница 0.046"}'

        res = run_json_agent("Посчитай разницу", max_iterations=6, llm_fn=mock_llm)
        self.assertEqual(res.approach, "json")
        self.assertEqual(res.answer, "Разница 0.046")
        self.assertEqual([s.tool for s in res.steps], ["parse_error", "lookup", "author_metrics", "calc"])
        self.assertIn("неизвестный инструмент", res.steps[1].result)

        # Проверка остановки по max_iterations
        loop_res = run_json_agent("Зацикливание", max_iterations=3, llm_fn=lambda _p: '{"action":"calc","input":"1+1"}')
        self.assertEqual(loop_res.iterations, 3)
        self.assertIn("Не удалось получить ответ", loop_res.answer)

    def test_native_agent_multistep_tool_call_id_and_limit(self) -> None:
        seen_messages: list[list[dict[str, Any]]] = []

        def mock_native(messages: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> LLMChatResult:
            self.assertEqual(len(schemas), 4)
            seen_messages.append(list(messages))
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            if len(tool_msgs) == 0:
                return LLMChatResult(
                    tool_calls=[
                        ToolCall(id="c1", name="classify_style", arguments={"text": "Текст о Петербурге"}),
                        ToolCall(id="c2", name="unknown_tool", arguments={"x": "1"}),
                    ]
                )
            if len(tool_msgs) == 2:
                return LLMChatResult(
                    tool_calls=[ToolCall(id="c3", name="calc", arguments={"expression": "10 - 3"})]
                )
            return LLMChatResult(content="Готово: Чехов А, разница 7")

        res = run_native_agent("Классифицируй и посчитай", max_iterations=5, llm_fn=mock_native)
        self.assertEqual(res.approach, "native")
        self.assertEqual(res.answer, "Готово: Чехов А, разница 7")
        self.assertEqual([s.tool for s in res.steps], ["classify_style", "unknown_tool", "calc"])

        # Проверяем, что tool_call_id и role="tool" корректно добавляются в историю
        last_msgs = seen_messages[-1]
        tool_ids = [m["tool_call_id"] for m in last_msgs if m.get("role") == "tool"]
        self.assertEqual(tool_ids, ["c1", "c2", "c3"])

        # Лимит итераций
        inf_res = run_native_agent(
            "Бесконечный цикл",
            max_iterations=2,
            llm_fn=lambda _m, _s: LLMChatResult(tool_calls=[ToolCall(id="x", name="calc", arguments={"expression": "1+1"})]),
        )
        self.assertEqual(inf_res.iterations, 2)
        self.assertIn("Не удалось получить ответ", inf_res.answer)

    def test_framework_agent_multistep_and_recursion_limit(self) -> None:
        res = run_framework_agent(
            "Составной запрос",
            max_iterations=6,
            chat_model=MockCompositeChatModel(),
        )
        self.assertEqual(res.approach, "framework")
        self.assertIn("Чехов А", res.answer)
        self.assertEqual([s.tool for s in res.steps], ["classify_style", "author_metrics", "author_metrics", "calc"])

        lim_res = run_framework_agent("Петля", max_iterations=2, chat_model=LoopingChatModel())
        self.assertIn("Не удалось получить ответ", lim_res.answer)


class LLMChatAndAgentEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_client_chat_parses_tool_calls(self) -> None:
        captured: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {"name": "calc", "arguments": '{"expression": "2+2"}'},
                                    }
                                ],
                            }
                        }
                    ]
                },
            )

        settings = Settings(llm_api_key="gsk_test", _env_file=None)
        client = LLMClient(settings)
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url=settings.resolved_base_url,
            transport=httpx.MockTransport(handler),
        )
        result = await client.chat([{"role": "user", "content": "2+2?"}], tools=TOOL_SCHEMAS)
        await client.aclose()

        self.assertIn("tools", captured["body"])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].id, "call_abc")
        self.assertEqual(result.tool_calls[0].name, "calc")
        self.assertEqual(result.tool_calls[0].arguments, {"expression": "2+2"})


class AgentFastAPIEndpointTests(unittest.TestCase):
    def test_post_agent_approaches_and_errors(self) -> None:
        dummy = DummyAgentClassifier()

        class MockAgentLLM:
            def __init__(self, *_a: object, **_kw: object) -> None:
                self.aclose = AsyncMock()
                self.generate = AsyncMock(return_value="Аналитический ответ по базе сюжетов.")

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                temperature: float = 0.0,
            ) -> LLMChatResult:
                if tools:
                    tool_msgs = [m for m in messages if m.get("role") == "tool"]
                    if not tool_msgs:
                        return LLMChatResult(
                            tool_calls=[ToolCall(id="t1", name="calc", arguments={"expression": "6+4"})]
                        )
                    return LLMChatResult(content=f"Итог native: {tool_msgs[0]['content']}")
                user_msg = messages[-1]["content"]
                if "Действие: calc" not in user_msg:
                    return LLMChatResult(content='{"action": "calc", "input": "6+4"}')
                return LLMChatResult(content='{"action": "final", "answer": "Итог json: 10"}')

        with (
            patch("src.app.load_classifier", return_value=dummy),
            patch("src.app.LLMClient", return_value=MockAgentLLM()),
            patch("src.agents.build_langchain_chat_model", return_value=MockCompositeChatModel()),
            TestClient(app) as client,
        ):
            r_nat = client.post("/agent", json={"question": "Сложи 6 и 4", "approach": "native"})
            self.assertEqual(r_nat.status_code, 200)
            self.assertEqual(r_nat.json()["approach"], "native")
            self.assertEqual(r_nat.json()["answer"], "Итог native: 10")

            r_json = client.post("/agent", json={"question": "Сложи 6 и 4", "approach": "json"})
            self.assertEqual(r_json.status_code, 200)
            self.assertEqual(r_json.json()["approach"], "json")
            self.assertEqual(r_json.json()["answer"], "Итог json: 10")

            r_fw = client.post("/agent", json={"question": "Составной запрос", "approach": "framework"})
            self.assertEqual(r_fw.status_code, 200)
            self.assertEqual(r_fw.json()["approach"], "framework")
            self.assertGreaterEqual(len(r_fw.json()["steps"]), 2)

            r_analyst = client.post(
                "/agent", json={"question": "Сравни средний рейтинг фантастики", "mode": "Аналитик"}
            )
            self.assertEqual(r_analyst.status_code, 200)
            self.assertEqual(r_analyst.json()["approach"], "analyst")
            self.assertGreaterEqual(len(r_analyst.json()["steps"]), 2)

            r_critic = client.post(
                "/agent", json={"question": "Посоветуй мистический сюжет", "mode": "Критик"}
            )
            self.assertEqual(r_critic.status_code, 200)
            self.assertEqual(r_critic.json()["approach"], "critic")
            self.assertGreaterEqual(len(r_critic.json()["steps"]), 2)

            # Валидация пустого ввода и неверного подхода
            self.assertEqual(client.post("/agent", json={"question": "   "}).status_code, 400)
            self.assertEqual(
                client.post("/agent", json={"question": "Привет", "approach": "invalid"}).status_code,
                422,
            )

            # Проброс LLMError -> 502
            with patch("src.app.run_native_agent_async", side_effect=LLMError("API error", 502)):
                r_err = client.post("/agent", json={"question": "Сложи 6 и 4", "approach": "native"})
                self.assertEqual(r_err.status_code, 502)


if __name__ == "__main__":
    unittest.main()
