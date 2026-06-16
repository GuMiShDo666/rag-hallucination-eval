"""Question rewriting for retrieval-oriented RAG queries."""

from __future__ import annotations

from dataclasses import dataclass
import os

from src.generator import PROVIDER_CONFIG


QUERY_REWRITE_PROMPT = """Rewrite the user question into a concise search query for document retrieval.

Rules:
1. Preserve the user's intent.
2. Keep important entities and technical terms.
3. Do not answer the question.
4. Return only the rewritten query.

Question:
{question}

Rewritten query:"""


@dataclass
class QueryRewriteResult:
    """A rewritten retrieval query and whether the original text changed."""

    original_query: str
    rewritten_query: str
    changed: bool
    model: str


class QueryRewriter:
    """Rewrite natural-language questions into retrieval-friendly queries."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        mock_mode: bool | None = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        self.mock_mode = _env_bool("MOCK_LLM") if mock_mode is None else mock_mode

        if self.provider == "mock":
            self.mock_mode = True

        provider_config = PROVIDER_CONFIG.get(self.provider)
        if provider_config:
            self.model = model or os.getenv(
                provider_config["model_env"],
                provider_config["default_model"],
            )
        else:
            self.model = model or "mock-model"

    def rewrite(self, question: str) -> QueryRewriteResult:
        """Return a retrieval query. Fall back to the original question when needed."""

        original = question.strip()
        if not original:
            return QueryRewriteResult(original, original, False, self.model)

        rewritten = self._rewrite_mock(original) if self.mock_mode else self._rewrite_with_provider(original)
        cleaned = _clean_rewritten_query(rewritten) or original
        return QueryRewriteResult(
            original_query=original,
            rewritten_query=cleaned,
            changed=cleaned != original,
            model=self.model,
        )

    def _rewrite_mock(self, question: str) -> str:
        return question

    def _rewrite_with_provider(self, question: str) -> str:
        provider_config = PROVIDER_CONFIG.get(self.provider)
        if not provider_config:
            return question

        api_key_env = provider_config["api_key_env"]
        api_key = os.getenv(api_key_env)
        if not api_key or api_key == "your_key":
            return question

        try:
            from openai import OpenAI
        except ImportError:
            return question

        client_kwargs = {"api_key": api_key}
        if provider_config["base_url"]:
            client_kwargs["base_url"] = provider_config["base_url"]

        try:
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": QUERY_REWRITE_PROMPT.format(question=question)}],
                temperature=0,
                max_tokens=64,
            )
        except Exception:
            return question

        content = response.choices[0].message.content if response.choices else None
        return content or question


def _clean_rewritten_query(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    prefixes = [
        "Rewritten query:",
        "Query:",
        "Search query:",
    ]
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def _env_bool(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}
