"""Project configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the RAG hallucination evaluation pipeline."""

    llm_provider: str = "qwen"
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    qwen_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    deepseek_model: str = "deepseek-chat"
    qwen_model: str = "qwen-plus"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    vector_store_path: str = "data/processed/faiss_index"
    default_chunk_size: int = 512
    default_chunk_overlap: int = 80
    default_top_k: int = 5
    mock_llm: bool = False

    @property
    def active_model(self) -> str:
        """Return the model name for the selected LLM provider."""

        if self.llm_provider == "deepseek":
            return self.deepseek_model
        if self.llm_provider == "qwen":
            return self.qwen_model
        return self.openai_model


def load_settings(env_path: str | None = None) -> Settings:
    """Load settings from `.env` and process environment variables."""

    load_dotenv(env_path or _default_env_path(), override=False)
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "qwen").lower(),
        openai_api_key=_clean_key(os.getenv("OPENAI_API_KEY")),
        deepseek_api_key=_clean_key(os.getenv("DEEPSEEK_API_KEY")),
        qwen_api_key=_clean_key(os.getenv("QWEN_API_KEY")),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        qwen_model=os.getenv("QWEN_MODEL", "qwen-plus"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        vector_store_path=os.getenv("VECTOR_STORE_PATH", "data/processed/faiss_index"),
        default_chunk_size=_env_int("DEFAULT_CHUNK_SIZE", 512),
        default_chunk_overlap=_env_int("DEFAULT_CHUNK_OVERLAP", 80),
        default_top_k=_env_int("DEFAULT_TOP_K", 5),
        mock_llm=_env_bool("MOCK_LLM"),
    )


def _default_env_path() -> str:
    return str(Path.cwd() / ".env")


def _clean_key(value: str | None) -> str | None:
    if not value or value == "your_key":
        return None
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer.") from None


def _env_bool(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}
