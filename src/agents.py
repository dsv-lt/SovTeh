"""Автономные ИИ-агенты с инструментами: Native Tool Calling, JSON-цикл и LangGraph.

Реализует Задание №2 (MLOps & AI-агенты):
1. Три доменных инструмента с docstring-контрактами и Pydantic-валидацией:
   - ``classify_style`` — локальная нейросетевая атрибуция стиля (64 русских классика);
   - ``author_metrics`` — поиск метрик (precision, recall, F1, support) из отчётов проекта;
   - ``calc`` — безопасный AST-калькулятор для арифметических вычислений;
2. Безопасная обёртка ``run_tool(name, args)`` с перехватом ошибок без обрыва цикла;
3. Три архитектуры агентного цикла:
   - ``run_native_agent`` — ручной цикл с OpenAI-совместимыми ``TOOL_SCHEMAS`` и ``tool_call_id``;
   - ``run_json_agent`` — ручной JSON-цикл с ``extract_json`` для любой LLM;
   - ``run_framework_agent`` — оркестрация через ``langgraph.prebuilt.create_react_agent``;
4. Наблюдаемость: пошаговая трассировка в лог/результат и опциональный MLflow Tracing.
"""

from __future__ import annotations

import ast
import asyncio
import csv
import inspect
import json
import logging
import operator
import re
from collections.abc import Callable
from contextlib import nullcontext
from functools import lru_cache
from typing import Any

import mlflow
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage as LangChainToolMessage
from langchain_core.tools import tool as lc_tool
from mlflow.entities import SpanType
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.classifier import BaselineClassifier, StyleClassifier, load_classifier
from src.config import PROJECT_ROOT, Settings, get_settings
from src.llm_client import LLMChatResult, LLMClient, LLMError
from src.preprocessing import enforce_network_immunity
from src.schemas import AgentStepTrace

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
LABELS_PATH = PROJECT_ROOT / "models" / "lora_classifier" / "labels.json"

_ACTIVE_CLASSIFIER: StyleClassifier | None = None

_ALLOWED_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# ---------------------------------------------------------------------------
# Состояние локального классификатора для инструмента classify_style
# ---------------------------------------------------------------------------

def set_agent_classifier(classifier: StyleClassifier | None) -> None:
    """Регистрирует предзагруженный классификатор (например, из lifespan FastAPI)."""
    global _ACTIVE_CLASSIFIER
    _ACTIVE_CLASSIFIER = classifier


def _get_classifier() -> StyleClassifier:
    """Возвращает активный классификатор или лениво загружает локальную модель."""
    global _ACTIVE_CLASSIFIER
    if _ACTIVE_CLASSIFIER is not None:
        return _ACTIVE_CLASSIFIER

    settings = get_settings()
    try:
        _ACTIVE_CLASSIFIER = load_classifier(settings)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Не удалось загрузить %s (%s); переключение на BaselineClassifier.",
            settings.classifier_backend,
            exc,
        )
        _ACTIVE_CLASSIFIER = BaselineClassifier(settings)
    return _ACTIVE_CLASSIFIER


# ---------------------------------------------------------------------------
# Инструменты (Шаг 1) и их Pydantic-контракты
# ---------------------------------------------------------------------------

class ClassifyStyleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, description="Фрагмент художественного текста на русском")


class AuthorMetricsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    author_query: str = Field(
        ...,
        min_length=1,
        description="Фамилия/имя автора (например, 'Чехов А') или 'overall' для сводных метрик",
    )


class CalcArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(
        ...,
        min_length=1,
        description="Арифметическое выражение из чисел и операторов + - * / ( )",
    )


class SearchPlotsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(
        ...,
        min_length=1,
        description="Поисковый запрос или тема сюжета (например, 'космическая экспедиция к Марсу')",
    )


def tool_classify_style(text: str) -> str:
    """Определить наиболее вероятного русского писателя-классика по фрагменту текста.

    Используй, когда нужно классифицировать стиль отрывка или узнать уверенность модели
    по авторам. Не используй для поиска метрик из отчётов или арифметики.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Текст для классификации не может быть пустым")
    clf = _get_classifier()
    pred = clf.predict(cleaned, k=3)
    payload = {
        "predicted_author": pred.author,
        "confidence": round(pred.confidence, 4),
        "model_name": clf.name,
        "top_3": [{"author": a, "probability": round(p, 4)} for a, p in pred.top_k],
    }
    return json.dumps(payload, ensure_ascii=False)


@lru_cache(maxsize=1)
def _load_reports_cache() -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Кэширует список 64 авторов, метрики по классам и общие сводки из reports/."""
    classes: list[str] = []
    if LABELS_PATH.exists():
        meta = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        classes = list(meta.get("classes", []))

    per_class: dict[str, dict[str, Any]] = {}
    csv_path = REPORTS_DIR / "lora_per_class_report.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                author = (row.get("") or "").strip()
                if not author:
                    continue
                per_class[author] = {
                    "author": author,
                    "precision": round(float(row["precision"]), 4),
                    "recall": round(float(row["recall"]), 4),
                    "f1_score": round(float(row["f1-score"]), 4),
                    "support": int(float(row["support"])),
                }

    summary: dict[str, Any] = {}
    for key, filename in (("lora", "metrics_lora.json"), ("baseline", "metrics_baseline.json")):
        p = REPORTS_DIR / filename
        if p.exists():
            summary[key] = json.loads(p.read_text(encoding="utf-8"))

    return classes, per_class, summary


def tool_author_metrics(author_query: str) -> str:
    """Получить тестовые метрики качества (precision, recall, f1_score, support) по автору.

    Используй, когда нужно узнать метрики классификатора для конкретного автора
    (например, 'Чехов А' или 'Достоевский Ф') или сводные метрики моделей ('overall').
    Не используй для классификации нового текста или арифметических вычислений.
    """
    query = author_query.strip()
    if not query:
        raise ValueError("Запрос автора не может быть пустым")

    classes, per_class, summary = _load_reports_cache()
    q_low = query.lower()

    if q_low in {"overall", "summary", "all", "итоги", "сводка", "общие"}:
        compact = {
            name: {
                "model": info.get("model"),
                "test_chunks": info.get("official_test_chunks"),
                "test_docs": info.get("official_test_docs"),
            }
            for name, info in summary.items()
        }
        return json.dumps(compact, ensure_ascii=False)

    exact = [row for author, row in per_class.items() if author.lower() == q_low]
    matches = exact or [row for author, row in per_class.items() if q_low in author.lower()]
    if matches:
        result = matches[0] if len(matches) == 1 else {"matches": matches[:5]}
        return json.dumps(result, ensure_ascii=False)

    known_authors = [a for a in classes if q_low in a.lower()]
    if known_authors:
        return json.dumps(
            {
                "authors": known_authors,
                "note": "Автор входит в 64 класса модели, но отсутствовал в test-разбиении отчёта.",
            },
            ensure_ascii=False,
        )

    return f"Автор по запросу '{query}' не найден в реестре (64 классика)."


def _eval_ast_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_eval_ast_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0.0:
            raise ZeroDivisionError("деление на ноль")
        return _ALLOWED_BIN_OPS[type(node.op)](_eval_ast_node(node.left), right)
    raise ValueError("недопустимая операция в выражении")


def tool_calc(expression: str) -> str:
    """Вычислить арифметическое выражение, например '0.9394 - 0.8934' или '2*(3+4)'.

    Используй только для чисел и операторов + - * / ( ). Не используй для поиска по базе.
    """
    cleaned = expression.strip().replace(",", ".")
    if not cleaned:
        return "Ошибка калькулятора: пустое выражение"
    if re.search(r"[^0-9+\-*/().\s]", cleaned):
        return "Ошибка калькулятора: допустимы только числа и операторы + - * / ( )"
    try:
        tree = ast.parse(cleaned, mode="eval")
        value = _eval_ast_node(tree.body)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as exc:
        return f"Ошибка калькулятора: {exc}"

    rounded = round(value, 6)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def tool_search_plots(query: str) -> str:
    """Найти релевантные произведения и описания сюжетов в подготовленном датасете.

    Используй, когда нужно найти сюжет по теме, событию, персонажу, жанру или автору
    в локальной базе знаний (data/plots_dataset.csv). Не используй для расчётов.
    """
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Поисковый запрос не может быть пустым")
    from src.rag import rag_search

    genre, docs, _items, exact = rag_search(cleaned, k=3)
    return json.dumps(
        {
            "predicted_genre": genre,
            "exact_match": exact,
            "results": docs,
        },
        ensure_ascii=False,
    )


TOOLS: dict[str, Callable[..., str]] = {
    "classify_style": tool_classify_style,
    "search_plots": tool_search_plots,
    "author_metrics": tool_author_metrics,
    "calc": tool_calc,
}

TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "classify_style": ClassifyStyleArgs,
    "search_plots": SearchPlotsArgs,
    "author_metrics": AuthorMetricsArgs,
    "calc": CalcArgs,
}


def build_tool_schemas() -> list[dict[str, Any]]:
    """Генерирует OpenAI JSON Schema из функций TOOLS и Pydantic-моделей."""
    schemas: list[dict[str, Any]] = []
    for name, fn in TOOLS.items():
        model_cls = TOOL_ARG_MODELS[name]
        raw_schema = model_cls.model_json_schema()
        properties = {
            prop: {k: v for k, v in meta.items() if k != "title"}
            for prop, meta in raw_schema.get("properties", {}).items()
        }
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": inspect.getdoc(fn) or name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": raw_schema.get("required", []),
                    },
                },
            }
        )
    return schemas


TOOL_SCHEMAS: list[dict[str, Any]] = build_tool_schemas()


# ---------------------------------------------------------------------------
# Безопасная обёртка run_tool (Шаг 2) и парсер JSON (JSON-подход, Шаг 2)
# ---------------------------------------------------------------------------

def run_tool(name: str, args: dict[str, Any] | str) -> str:
    """Безопасно валидирует аргументы и вызывает инструмент, возвращая ошибку строкой."""
    clean_name = str(name).strip()
    if clean_name not in TOOLS:
        return f"Ошибка: неизвестный инструмент '{clean_name}'. Доступны: {list(TOOLS)}."

    model_cls = TOOL_ARG_MODELS[clean_name]
    field_names = list(model_cls.model_fields.keys())
    try:
        if isinstance(args, dict):
            if "_raw_invalid_json" in args:
                return f"Ошибка аргументов {clean_name}: невалидный JSON ({args['_raw_invalid_json']})"
            validated = model_cls.model_validate(args)
        else:
            if len(field_names) != 1:
                return f"Ошибка аргументов {clean_name}: ожидается словарь параметров {field_names}"
            validated = model_cls.model_validate({field_names[0]: str(args)})
        return str(TOOLS[clean_name](**validated.model_dump()))
    except ValidationError as exc:
        return f"Ошибка валидации аргументов {clean_name}: {exc.errors()[0].get('msg', exc)}"
    except Exception as exc:  # noqa: BLE001 — изоляция сбоя инструмента, чтобы агент продолжил цикл
        return f"Ошибка инструмента {clean_name}: {exc}"


def extract_json(text: str) -> dict[str, Any]:
    """Извлекает первый JSON-объект из текста или markdown-блока ``json``."""
    cleaned = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("В ответе модели нет JSON")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(cleaned[start:])
    if not isinstance(obj, dict):
        raise TypeError("Корневой JSON должен быть объектом")
    return obj


# ---------------------------------------------------------------------------
# Результат работы агента и вспомогательные утилиты
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """Итоговый результат прогона одного из трёх агентных подходов."""

    approach: str
    answer: str
    iterations: int
    steps: list[AgentStepTrace]
    trace_id: str | None = None


NATIVE_SYSTEM_PROMPT = (
    "Ты — интеллектуальный ассистент по анализу русского литературного стиля и сюжетов. "
    "У тебя есть инструменты: classify_style (определение автора по тексту), "
    "search_plots (семантический поиск сюжетов в подготовленном датасете), "
    "author_metrics (метрики авторов из отчётов) и calc (арифметический калькулятор). "
    "Вызывай необходимые инструменты последовательно, пока не соберёшь все точные факты, "
    "затем дай ясный финальный ответ на русском языке."
)

JSON_SYSTEM_PROMPT = (
    "Ты — интеллектуальный ассистент по анализу русского литературного стиля и сюжетов.\n"
    "Доступные инструменты:\n"
    "- classify_style(text) — определить вероятного автора по фрагменту текста;\n"
    "- search_plots(query) — найти произведения и сюжеты в базе данных по запросу;\n"
    "- author_metrics(author_query) — получить precision/recall/f1_score автора (или 'overall');\n"
    "- calc(expression) — вычислить арифметическое выражение.\n\n"
    "На каждом шаге верни ТОЛЬКО один JSON-объект без пояснений:\n"
    '{"action": "classify_style|search_plots|author_metrics|calc|final", "input": "...", "answer": "..."}\n'
    "Для action=final заполни поле answer, для вызова инструмента — поле input.\n"
    "Используй инструменты, пока не соберёшь все данные для ответа."
)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _trace_context(
    enable_mlflow: bool,
    name: str,
    span_type: str = SpanType.UNKNOWN,
) -> Any:
    if not enable_mlflow:
        return nullcontext(None)
    return mlflow.start_span(name=name, span_type=span_type)


def _execute_traced_tool(name: str, args: dict[str, Any] | str, enable_mlflow: bool) -> str:
    """Выполняет run_tool внутри дочернего спана MLflow при включённой трассировке."""
    with _trace_context(enable_mlflow, f"tool:{name}", span_type=SpanType.TOOL) as span:
        if span is not None:
            span.set_inputs({"name": name, "args": args})
        result = run_tool(name, args)
        if span is not None:
            span.set_outputs({"result": result})
        return result


# ---------------------------------------------------------------------------
# Подход 1: Native Tool Calling (ручной цикл со схемами TOOL_SCHEMAS)
# ---------------------------------------------------------------------------

async def run_native_agent_async(
    user_query: str,
    max_iterations: int = 6,
    llm_client: LLMClient | None = None,
    llm_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any] | None = None,
    temperature: float = 0.0,
    enable_mlflow_trace: bool = False,
) -> AgentResult:
    """Ручной агентный цикл через нативный tool calling (``tools=TOOL_SCHEMAS``)."""
    if llm_fn is None and llm_client is None:
        raise ValueError("Необходимо передать llm_client или llm_fn")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": NATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    steps: list[AgentStepTrace] = []
    answer = "Не удалось получить ответ за отведенное число шагов."
    iterations_used = max_iterations

    with _trace_context(enable_mlflow_trace, "native_agent", span_type=SpanType.AGENT) as root_span:
        if root_span is not None:
            root_span.set_inputs({"query": user_query})

        for step_num in range(1, max_iterations + 1):
            with _trace_context(enable_mlflow_trace, f"llm_step_{step_num}", span_type=SpanType.LLM) as llm_span:
                if llm_span is not None:
                    llm_span.set_inputs({"step": step_num, "messages_count": len(messages)})
                if llm_fn is not None:
                    response: LLMChatResult = await _maybe_await(llm_fn(messages, TOOL_SCHEMAS))
                else:
                    assert llm_client is not None
                    response = await llm_client.chat(messages, tools=TOOL_SCHEMAS, temperature=temperature)
                if llm_span is not None:
                    llm_span.set_outputs(
                        {
                            "content": response.content,
                            "tool_calls": [c.name for c in response.tool_calls],
                        }
                    )

            if not response.tool_calls:
                answer = response.content or "Ответ не сформирован."
                iterations_used = step_num
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
            )

            for call in response.tool_calls:
                result = _execute_traced_tool(call.name, call.arguments, enable_mlflow_trace)
                logger.info(
                    "[native][step %d] tool=%s args=%s -> %s",
                    step_num,
                    call.name,
                    call.arguments,
                    result[:140],
                )
                steps.append(
                    AgentStepTrace(
                        step=step_num,
                        tool=call.name,
                        args=call.arguments,
                        result=result,
                    )
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        if root_span is not None:
            root_span.set_outputs({"answer": answer, "steps": len(steps)})

    trace_id = mlflow.get_last_active_trace_id() if enable_mlflow_trace else None
    return AgentResult(
        approach="native",
        answer=answer,
        iterations=iterations_used,
        steps=steps,
        trace_id=trace_id,
    )


def run_native_agent(
    user_query: str,
    max_iterations: int = 6,
    llm_client: LLMClient | None = None,
    llm_fn: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any] | None = None,
    temperature: float = 0.0,
    enable_mlflow_trace: bool = False,
) -> AgentResult:
    """Синхронная обёртка над ``run_native_agent_async`` для CLI и автотестов."""
    return asyncio.run(
        run_native_agent_async(
            user_query=user_query,
            max_iterations=max_iterations,
            llm_client=llm_client,
            llm_fn=llm_fn,
            temperature=temperature,
            enable_mlflow_trace=enable_mlflow_trace,
        )
    )


# ---------------------------------------------------------------------------
# Подход 2: Альтернативный JSON-подход для любой LLM (ручной парсинг JSON)
# ---------------------------------------------------------------------------

async def run_json_agent_async(
    user_query: str,
    max_iterations: int = 6,
    llm_client: LLMClient | None = None,
    llm_fn: Callable[[str], Any] | None = None,
    temperature: float = 0.0,
    enable_mlflow_trace: bool = False,
) -> AgentResult:
    """Ручной агентный цикл через строгий JSON-формат без native tool calling."""
    if llm_fn is None and llm_client is None:
        raise ValueError("Необходимо передать llm_client или llm_fn")

    history = f"Вопрос: {user_query}\n"
    steps: list[AgentStepTrace] = []
    answer = "Не удалось получить ответ за отведенное число шагов."
    iterations_used = max_iterations

    with _trace_context(enable_mlflow_trace, "json_agent", span_type=SpanType.AGENT) as root_span:
        if root_span is not None:
            root_span.set_inputs({"query": user_query})

        for step_num in range(1, max_iterations + 1):
            with _trace_context(enable_mlflow_trace, f"llm_step_{step_num}", span_type=SpanType.LLM) as llm_span:
                if llm_span is not None:
                    llm_span.set_inputs({"step": step_num, "history_len": len(history)})
                if llm_fn is not None:
                    raw = str(await _maybe_await(llm_fn(f"{JSON_SYSTEM_PROMPT}\n{history}")))
                else:
                    assert llm_client is not None
                    res = await llm_client.chat(
                        messages=[
                            {"role": "system", "content": JSON_SYSTEM_PROMPT},
                            {"role": "user", "content": history},
                        ],
                        tools=None,
                        temperature=temperature,
                    )
                    raw = res.content or ""
                if llm_span is not None:
                    llm_span.set_outputs({"raw": raw[:300]})

            logger.info("[json][step %d] raw=%s", step_num, raw[:200].replace("\n", " "))
            try:
                parsed = extract_json(raw)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                err_msg = f"Ошибка: ответ не парсится как JSON ({exc}). Верни чистый JSON."
                steps.append(
                    AgentStepTrace(
                        step=step_num,
                        tool="parse_error",
                        args=None,
                        result=err_msg,
                        raw=raw[:300],
                    )
                )
                history += f"{err_msg}\n"
                continue

            action = str(parsed.get("action") or parsed.get("tool") or "").strip().lower()
            if action == "final":
                answer = str(parsed.get("answer") or "Модель не дала ответ")
                iterations_used = step_num
                break

            input_value = parsed.get("input", parsed.get("arg", ""))
            tool_args = input_value if isinstance(input_value, dict) else str(input_value)
            result = _execute_traced_tool(action, tool_args, enable_mlflow_trace)
            logger.info(
                "[json][step %d] action=%s input=%s -> %s",
                step_num,
                action,
                input_value,
                result[:140],
            )
            steps.append(
                AgentStepTrace(
                    step=step_num,
                    tool=action,
                    args=tool_args,
                    result=result,
                    raw=raw[:300],
                )
            )
            history += f"Действие: {action}({input_value}) -> {result}\n"

        if root_span is not None:
            root_span.set_outputs({"answer": answer, "steps": len(steps)})

    trace_id = mlflow.get_last_active_trace_id() if enable_mlflow_trace else None
    return AgentResult(
        approach="json",
        answer=answer,
        iterations=iterations_used,
        steps=steps,
        trace_id=trace_id,
    )


def run_json_agent(
    user_query: str,
    max_iterations: int = 6,
    llm_client: LLMClient | None = None,
    llm_fn: Callable[[str], Any] | None = None,
    temperature: float = 0.0,
    enable_mlflow_trace: bool = False,
) -> AgentResult:
    """Синхронная обёртка над ``run_json_agent_async`` для CLI и автотестов."""
    return asyncio.run(
        run_json_agent_async(
            user_query=user_query,
            max_iterations=max_iterations,
            llm_client=llm_client,
            llm_fn=llm_fn,
            temperature=temperature,
            enable_mlflow_trace=enable_mlflow_trace,
        )
    )


# ---------------------------------------------------------------------------
# Подход 3: Фреймворк (LangGraph create_react_agent)
# ---------------------------------------------------------------------------

def build_langchain_chat_model(settings: Settings | None = None, temperature: float = 0.0) -> Any:
    """Создаёт ``ChatOpenAI`` на базе конфигурации провайдера из ``Settings``."""
    cfg = settings or get_settings()
    if cfg.bypass_system_proxy:
        enforce_network_immunity()
    api_key = cfg.effective_api_key
    if not api_key or not api_key.get_secret_value().strip():
        raise LLMError("LLM_API_KEY не задан: передайте ключ через переменную окружения", 503)

    import httpx
    from langchain_openai import ChatOpenAI

    headers = {"X-Title": "Literary Style Assistant"} if cfg.resolved_provider == "openrouter" else None
    http_client = httpx.Client(
        timeout=cfg.llm_timeout,
        trust_env=not cfg.bypass_system_proxy,
        verify=cfg.resolved_provider != "gigachat",
    )
    http_async_client = httpx.AsyncClient(
        timeout=cfg.llm_timeout,
        trust_env=not cfg.bypass_system_proxy,
        verify=cfg.resolved_provider != "gigachat",
    )
    return ChatOpenAI(
        api_key=api_key.get_secret_value().strip(),
        base_url=cfg.resolved_base_url.rstrip("/"),
        model=cfg.resolved_model,
        temperature=temperature,
        timeout=cfg.llm_timeout,
        max_retries=cfg.llm_max_retries,
        default_headers=headers,
        http_client=http_client,
        http_async_client=http_async_client,
    )


def _make_langgraph_tools() -> list[Any]:
    @lc_tool
    def classify_style(text: str) -> str:
        """Определить наиболее вероятного русского писателя-классика по фрагменту текста."""
        return run_tool("classify_style", {"text": text})

    @lc_tool
    def search_plots(query: str) -> str:
        """Найти произведения и описания сюжетов в локальном датасете по запросу."""
        return run_tool("search_plots", {"query": query})

    @lc_tool
    def author_metrics(author_query: str) -> str:
        """Получить тестовые метрики (precision, recall, f1_score, support) по автору или 'overall'."""
        return run_tool("author_metrics", {"author_query": author_query})

    @lc_tool
    def calc(expression: str) -> str:
        """Вычислить арифметическое выражение, например '0.9394 - 0.8934'."""
        return run_tool("calc", {"expression": expression})

    return [classify_style, search_plots, author_metrics, calc]


def _extract_framework_steps(messages: list[Any]) -> list[AgentStepTrace]:
    tool_outputs: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, LangChainToolMessage) and getattr(msg, "tool_call_id", None):
            tool_outputs[str(msg.tool_call_id)] = str(msg.content)

    steps: list[AgentStepTrace] = []
    step_idx = 0
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            step_idx += 1
            tc_id = str(tc.get("id") or "")
            name = str(tc.get("name") or "")
            args = tc.get("args") or {}
            res = tool_outputs.get(tc_id, "")
            logger.info("[framework][step %d] tool=%s args=%s -> %s", step_idx, name, args, res[:140])
            steps.append(AgentStepTrace(step=step_idx, tool=name, args=args, result=res))
    return steps


async def run_framework_agent_async(
    user_query: str,
    max_iterations: int = 6,
    settings: Settings | None = None,
    chat_model: Any | None = None,
    temperature: float = 0.0,
    enable_mlflow_trace: bool = False,
) -> AgentResult:
    """Агентная оркестрация через ``langgraph.prebuilt.create_react_agent``."""
    from langgraph.errors import GraphRecursionError
    from langgraph.prebuilt import create_react_agent

    if enable_mlflow_trace:
        mlflow.langchain.autolog()

    model = chat_model or build_langchain_chat_model(settings=settings, temperature=temperature)
    graph = create_react_agent(model, _make_langgraph_tools(), prompt=NATIVE_SYSTEM_PROMPT)

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_query)]},
            config={"recursion_limit": max_iterations * 2 + 2},
        )
    except GraphRecursionError:
        trace_id = mlflow.get_last_active_trace_id() if enable_mlflow_trace else None
        return AgentResult(
            approach="framework",
            answer="Не удалось получить ответ за отведенное число шагов.",
            iterations=max_iterations,
            steps=[],
            trace_id=trace_id,
        )

    all_msgs = result.get("messages", [])
    steps = _extract_framework_steps(all_msgs)
    raw_answer = str(getattr(all_msgs[-1], "content", "") or "Ответ не сформирован.") if all_msgs else ""
    if "need more steps to process this request" in raw_answer.lower():
        answer = "Не удалось получить ответ за отведенное число шагов."
    else:
        answer = raw_answer
    trace_id = mlflow.get_last_active_trace_id() if enable_mlflow_trace else None
    return AgentResult(
        approach="framework",
        answer=answer,
        iterations=max(1, len(steps) + 1),
        steps=steps,
        trace_id=trace_id,
    )


def run_framework_agent(
    user_query: str,
    max_iterations: int = 6,
    settings: Settings | None = None,
    chat_model: Any | None = None,
    temperature: float = 0.0,
    enable_mlflow_trace: bool = False,
) -> AgentResult:
    """Синхронная обёртка над ``run_framework_agent_async`` для CLI и автотестов."""
    return asyncio.run(
        run_framework_agent_async(
            user_query=user_query,
            max_iterations=max_iterations,
            settings=settings,
            chat_model=chat_model,
            temperature=temperature,
            enable_mlflow_trace=enable_mlflow_trace,
        )
    )
