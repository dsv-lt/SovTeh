"""Сравнение трёх архитектур ИИ-агентов (Native Tool Calling, JSON-цикл, LangGraph).

Реализует Задание №2 (MLOps & AI-агенты):
- составной запрос, требующий вызова минимум 2 инструментов (``classify_style`` / ``author_metrics`` / ``calc``);
- пошаговая трассировка вызовов инструментов (и опциональный MLflow Tracing);
- демонстрация как с реальным LLM API (из ``.env``), так и в автономном ``--mock`` режиме.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

import mlflow
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages import ToolMessage as LCToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.agents import (
    AgentResult,
    run_framework_agent,
    run_framework_agent_async,
    run_json_agent,
    run_json_agent_async,
    run_native_agent,
    run_native_agent_async,
    set_agent_classifier,
)
from src.classifier import BaselineClassifier
from src.config import PROJECT_ROOT, get_settings
from src.llm_client import LLMChatResult, LLMClient, ToolCall

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DEFAULT_COMPOSITE_QUERY = (
    "Определи автора отрывка «В саду за старой усадьбой тихо осыпались липы; "
    "доктор пил остывший чай на террасе», узнай F1-score для найденного автора и для "
    "'Достоевский Ф', а затем вычисли точную разницу между их F1-score через калькулятор."
)


class MockCompositeChatModel(BaseChatModel):
    """Детерминированная LangChain-модель с bind_tools для демонстрации LangGraph без API-ключа."""

    @property
    def _llm_type(self) -> str:
        return "mock-composite-chat"

    def bind_tools(self, tools: Any, **_kwargs: Any) -> MockCompositeChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_msgs = [m for m in messages if isinstance(m, LCToolMessage)]
        if len(tool_msgs) == 0:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "fw_1",
                        "name": "classify_style",
                        "args": {"text": "В саду за старой усадьбой тихо осыпались липы; доктор пил остывший чай на террасе"},
                    }
                ],
            )
        elif len(tool_msgs) == 1:
            detected = json.loads(tool_msgs[0].content).get("predicted_author", "Чехов А")
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"id": "fw_2", "name": "author_metrics", "args": {"author_query": detected}},
                    {"id": "fw_3", "name": "author_metrics", "args": {"author_query": "Достоевский Ф"}},
                ],
            )
        elif len(tool_msgs) == 3:
            f1_a = json.loads(tool_msgs[1].content).get("f1_score", 0.8934)
            f1_b = json.loads(tool_msgs[2].content).get("f1_score", 0.9394)
            msg = AIMessage(
                content="",
                tool_calls=[{"id": "fw_4", "name": "calc", "args": {"expression": f"{f1_b} - {f1_a}"}}],
            )
        else:
            diff = tool_msgs[-1].content
            author_a = json.loads(tool_msgs[0].content).get("predicted_author", "Чехов А")
            msg = AIMessage(
                content=f"Стиль отрывка определён как {author_a}; разница F1 с Достоевским Ф равна {diff}."
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _make_mock_native_llm() -> Any:
    """Замоканная LLM для демонстрации Native Tool Calling."""
    def mock_native(messages: list[dict[str, Any]], _schemas: list[dict[str, Any]]) -> LLMChatResult:
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if len(tool_msgs) == 0:
            return LLMChatResult(
                tool_calls=[
                    ToolCall(
                        id="nat_1",
                        name="classify_style",
                        arguments={
                            "text": "В саду за старой усадьбой тихо осыпались липы; доктор пил остывший чай на террасе"
                        },
                    )
                ]
            )
        if len(tool_msgs) == 1:
            detected = json.loads(tool_msgs[0]["content"]).get("predicted_author", "Чехов А")
            return LLMChatResult(
                tool_calls=[
                    ToolCall(id="nat_2", name="author_metrics", arguments={"author_query": detected}),
                    ToolCall(id="nat_3", name="author_metrics", arguments={"author_query": "Достоевский Ф"}),
                ]
            )
        if len(tool_msgs) == 3:
            f1_a = json.loads(tool_msgs[1]["content"]).get("f1_score", 0.8934)
            f1_b = json.loads(tool_msgs[2]["content"]).get("f1_score", 0.9394)
            return LLMChatResult(
                tool_calls=[ToolCall(id="nat_4", name="calc", arguments={"expression": f"{f1_b} - {f1_a}"})]
            )
        author_a = json.loads(tool_msgs[0]["content"]).get("predicted_author", "Чехов А")
        diff = tool_msgs[-1]["content"]
        return LLMChatResult(
            content=f"Стиль отрывка определён как {author_a}; разница F1 с Достоевским Ф равна {diff}."
        )

    return mock_native


def _make_mock_json_llm() -> Any:
    """Замоканная LLM для демонстрации JSON-подхода (включая unwrap markdown-обёртки)."""
    state: dict[str, Any] = {"author": "Чехов А", "f1_a": 0.0, "f1_b": 0.0}

    def mock_json(prompt: str) -> str:
        lines = [ln for ln in prompt.splitlines() if ln.startswith("Действие:")]
        if len(lines) == 0:
            return (
                '```json\n{"action": "classify_style", "input": '
                '"В саду за старой усадьбой тихо осыпались липы; доктор пил остывший чай на террасе"}\n```'
            )
        if len(lines) == 1:
            payload = json.loads(lines[0].split("->", 1)[1].strip())
            state["author"] = payload.get("predicted_author", "Чехов А")
            return json.dumps({"action": "author_metrics", "input": state["author"]}, ensure_ascii=False)
        if len(lines) == 2:
            payload = json.loads(lines[1].split("->", 1)[1].strip())
            state["f1_a"] = payload.get("f1_score", 0.8934)
            return '{"action": "author_metrics", "input": "Достоевский Ф"}'
        if len(lines) == 3:
            payload = json.loads(lines[2].split("->", 1)[1].strip())
            state["f1_b"] = payload.get("f1_score", 0.9394)
            return json.dumps({"action": "calc", "input": f"{state['f1_b']} - {state['f1_a']}"}, ensure_ascii=False)
        diff = lines[-1].split("->", 1)[1].strip()
        return json.dumps(
            {
                "action": "final",
                "answer": f"Стиль отрывка определён как {state['author']}; разница F1 с Достоевским Ф равна {diff}.",
            },
            ensure_ascii=False,
        )

    return mock_json


def _print_result(res: AgentResult) -> None:
    print(f"\n{'=' * 68}")
    print(f"Подход: {res.approach.upper()} | Итераций: {res.iterations} | Вызовов инструментов: {len(res.steps)}")
    if res.trace_id:
        print(f"MLflow Trace ID: {res.trace_id}")
    print("-" * 68)
    for st in res.steps:
        print(f"  [step {st.step}] tool={st.tool} args={st.args} -> {st.result[:110]}")
    print("-" * 68)
    print(f"Итоговый ответ: {res.answer}")


def compare_all(query: str, use_mock: bool = False, trace_mlflow: bool = False) -> list[AgentResult]:
    """Запускает и сравнивает 3 подхода на одном составном запросе."""
    settings = get_settings()
    set_agent_classifier(BaselineClassifier(settings))

    if trace_mlflow:
        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("doc-classifier-agents")

    if use_mock:
        r_native = run_native_agent(query, llm_fn=_make_mock_native_llm(), enable_mlflow_trace=trace_mlflow)
        r_json = run_json_agent(query, llm_fn=_make_mock_json_llm(), enable_mlflow_trace=trace_mlflow)
        r_fw = run_framework_agent(
            query, chat_model=MockCompositeChatModel(), enable_mlflow_trace=trace_mlflow
        )
    else:
        async def _run_live() -> tuple[AgentResult, AgentResult, AgentResult]:
            client = LLMClient(settings)
            try:
                rn = await run_native_agent_async(
                    query, llm_client=client, enable_mlflow_trace=trace_mlflow
                )
                rj = await run_json_agent_async(
                    query, llm_client=client, enable_mlflow_trace=trace_mlflow
                )
                rf = await run_framework_agent_async(
                    query, settings=settings, enable_mlflow_trace=trace_mlflow
                )
                return rn, rj, rf
            finally:
                await client.aclose()

        r_native, r_json, r_fw = asyncio.run(_run_live())

    if trace_mlflow:
        mlflow.flush_trace_async_logging()

    results = [r_native, r_json, r_fw]
    for r in results:
        _print_result(r)

    print(f"\n{'=' * 68}")
    print("СВОДНОЕ СРАВНЕНИЕ ТРЁХ ПОДХОДОВ:")
    print(f"{'Подход':<12} | {'Итерации':<10} | {'Шаги tools':<12} | {'Инструменты'}")
    print("-" * 68)
    for r in results:
        tools_used = " -> ".join(s.tool for s in r.steps) or "-"
        print(f"{r.approach:<12} | {r.iterations:<10} | {len(r.steps):<12} | {tools_used}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение 3 архитектур ИИ-агентов (Задание №2)")
    parser.add_argument("--query", default=DEFAULT_COMPOSITE_QUERY, help="Составной запрос")
    parser.add_argument("--mock", action="store_true", help="Автономный режим с замоканной LLM")
    parser.add_argument("--trace", action="store_true", help="Сохранять MLflow Traces в sqlite:///mlflow.db")
    args = parser.parse_args()
    compare_all(query=args.query, use_mock=args.mock, trace_mlflow=args.trace)


if __name__ == "__main__":
    main()
