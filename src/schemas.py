"""Pydantic-схемы запросов и ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=15000,
        description="Запрос пользователя по сюжетам или фрагмент текста (до 15 000 символов)",
    )
    k: int = Field(default=3, ge=1, le=10, description="Количество сюжетов / top-k авторов")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class AuthorScore(BaseModel):
    author: str
    probability: float


class QueryResponse(BaseModel):
    answer: str = Field(default="", description="Контекстный ответ ИИ-агента по базе сюжетов")
    sources: list[str] = Field(
        default_factory=list,
        description="Релевантные сюжеты из локальной базы знаний (RAG)",
    )
    predicted_author: str = Field(..., description="Ближайший авторский стиль (локальный классификатор)")
    predicted_genre: str = Field(
        default="драма и мелодрама",
        description="Жанровая направленность (локальный ML-фильтр)",
    )
    confidence: float
    model_name: str
    generated_text: str


class ClassifyResponse(BaseModel):
    predicted_author: str
    confidence: float
    model_name: str
    top_authors: list[AuthorScore]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class AgentRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=15000,
        description="Составной запрос к ИИ-агенту (до 15 000 символов)",
    )
    approach: str = Field(
        default="native",
        pattern="^(native|json|framework)$",
        description="Архитектура агентного цикла: native, json или framework",
    )
    mode: str | None = Field(
        default=None,
        description="Опциональный режим мультиагентного интерфейса: RAG, Аналитик или Критик",
    )
    max_iterations: int = Field(default=6, ge=1, le=12, description="Максимум шагов цикла агента")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class AgentStepTrace(BaseModel):
    step: int
    tool: str
    args: dict | str | None = None
    result: str
    raw: str | None = None


class AgentResponse(BaseModel):
    approach: str
    answer: str
    iterations: int
    steps: list[AgentStepTrace | str]
