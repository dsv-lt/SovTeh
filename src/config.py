"""Конфигурация сервиса (pydantic-settings).

Все параметры читаются из переменных окружения (или файла ``.env``).
Секреты (``LLM_API_KEY``, ``OPENAI_API_KEY``, ``GEMINI_API_KEY``,
``GROQ_API_KEY``, ``OPENROUTER_API_KEY``, ``GIGACHAT_CREDENTIALS``)
никогда не хранятся в коде и образе Docker.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROVIDER_DEFAULTS: dict[str, dict[str, object]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "fallback_models": ("llama-3.3-70b-versatile",),
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "fallback_models": ("google/gemini-2.0-flash-exp:free",),
    },
    "aistudio": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.1-flash-lite",
        "fallback_models": ("gemini-flash-lite-latest", "gemma-4-31b-it"),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "fallback_models": ("gpt-4.1-mini", "gpt-3.5-turbo"),
    },
    "gigachat": {
        "base_url": "https://gigachat.devices.sberbank.ru/api/v1",
        "model": "GigaChat-2",
        "fallback_models": ("GigaChat-2-Lite", "GigaChat"),
    },
    "custom": {
        "base_url": "http://localhost:11434/v1",
        "model": "default",
        "fallback_models": (),
    },
}


class Settings(BaseSettings):
    """Настройки приложения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Внешняя LLM (любая модель по API) ---
    llm_api_key: SecretStr | None = Field(
        default=None,
        description="Универсальный ключ Groq / OpenRouter / Google AI Studio / OpenAI / GigaChat / Custom API",
    )
    openai_api_key: SecretStr | None = Field(default=None, description="Резервный ключ OpenAI")
    gemini_api_key: SecretStr | None = Field(default=None, description="Резервный ключ Google AI Studio")
    groq_api_key: SecretStr | None = Field(default=None, description="Резервный ключ Groq Cloud")
    openrouter_api_key: SecretStr | None = Field(default=None, description="Резервный ключ OpenRouter")
    gigachat_credentials: SecretStr | None = Field(default=None, description="Ключ авторизации Sber GigaChat")

    llm_provider: Literal["auto", "groq", "openrouter", "aistudio", "openai", "gigachat", "custom"] = Field(
        default="auto",
        description=(
            "auto: определяется по префиксу ключа или заданному URL "
            "(gsk_ -> Groq, sk-or- -> OpenRouter, AQ./AIza -> Google AI Studio, sk- -> OpenAI, "
            "GIGACHAT_CREDENTIALS -> GigaChat)"
        ),
    )
    llm_base_url: str | None = Field(default=None, description="Переопределение Base URL OpenAI-совместимого API")
    llm_model: str | None = Field(default=None, description="Переопределение имени модели")
    llm_timeout: float = Field(default=25.0, gt=0)
    llm_max_tokens: int = Field(default=1024, ge=16, le=8192)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_auto_discover_models: bool = Field(
        default=True,
        description="Динамически опрашивать доступные модели провайдера при старте (GET /models)",
    )
    bypass_system_proxy: bool = Field(
        default=True,
        description="Изолировать процесс от системных прокси-переменных Windows (trust_env=False)",
    )

    # --- Локальный классификатор и RAG (стадия 1 + семантический поиск) ---
    classifier_backend: Literal["lora", "baseline"] = "lora"
    models_dir: Path = PROJECT_ROOT / "models"
    dataset_path: Path = PROJECT_ROOT / "data" / "plots_dataset.csv"
    rag_embedding_backend: Literal["auto", "rubert", "hybrid"] = "auto"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    chunk_chars: int = Field(default=1000, ge=100)
    max_chunks: int = Field(default=16, ge=1, description="Максимум фрагментов на запрос")

    @property
    def vectorizer_path(self) -> Path:
        return self.models_dir / "tfidf_vectorizer.pkl"

    @property
    def label_encoder_path(self) -> Path:
        return self.models_dir / "label_encoder.pkl"

    @property
    def baseline_path(self) -> Path:
        return self.models_dir / "baseline_pytorch.pt"

    @property
    def lora_dir(self) -> Path:
        return self.models_dir / "lora_classifier"

    @property
    def base_model_dir(self) -> Path:
        return self.models_dir / "rubert_base"

    @property
    def genre_model_path(self) -> Path:
        return self.models_dir / "genre_classifier_model.pkl"

    @property
    def hashing_vectorizer_path(self) -> Path:
        return self.models_dir / "hashing_vectorizer.pkl"

    @property
    def effective_api_key(self) -> SecretStr | None:
        if "llm_api_key" in self.model_fields_set and self.llm_api_key is None:
            return None
        for candidate in (
            self.llm_api_key,
            self.gemini_api_key,
            self.groq_api_key,
            self.openrouter_api_key,
            self.openai_api_key,
            self.gigachat_credentials,
        ):
            if candidate is not None and candidate.get_secret_value().strip():
                return candidate
        return None

    @property
    def resolved_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        key_obj = self.effective_api_key
        key = key_obj.get_secret_value().strip() if key_obj else ""
        if key.startswith("sk-or-"):
            return "openrouter"
        if key.startswith(("AQ.", "AIza")):
            return "aistudio"
        if key.startswith("gsk_"):
            return "groq"
        if key.startswith("sk-"):
            return "openai"
        if (
            self.gigachat_credentials is not None
            and self.gigachat_credentials.get_secret_value().strip()
            and self.llm_api_key is None
        ):
            return "gigachat"
        if self.llm_base_url:
            return "custom"
        return "groq"

    @property
    def resolved_base_url(self) -> str:
        return self.llm_base_url or str(PROVIDER_DEFAULTS[self.resolved_provider]["base_url"])

    @property
    def resolved_model(self) -> str:
        return self.llm_model or str(PROVIDER_DEFAULTS[self.resolved_provider]["model"])

    @property
    def resolved_candidate_models(self) -> tuple[str, ...]:
        if self.llm_model:
            return (self.llm_model,)
        primary = self.resolved_model
        fallbacks = tuple(PROVIDER_DEFAULTS[self.resolved_provider].get("fallback_models", ()))  # type: ignore[arg-type]
        return (primary, *(m for m in fallbacks if m != primary))


@lru_cache
def get_settings() -> Settings:
    return Settings()
