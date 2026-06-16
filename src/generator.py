"""LLM answer generation for retrieved RAG contexts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

from src.retriever import RetrievedChunk


PROMPT_TEMPLATE = """You are a research assistant. Answer the question strictly based on the provided context.

Rules:
1. Use only the provided context.
2. If the context does not contain enough information, say: "The provided context is insufficient."
3. Cite the context using [1], [2], [3] when possible.
4. Do not add facts that are not supported by the context.

Question:
{question}

Context:
{context}

Answer:"""


PROVIDER_CONFIG = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
        "base_url": None,
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
    "qwen": {
        "api_key_env": "QWEN_API_KEY",
        "model_env": "QWEN_MODEL",
        "default_model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
}


@dataclass
class GenerationResult:
    """Generated answer with its prompt and source contexts."""

    question: str
    answer: str
    contexts: list[RetrievedChunk]
    prompt: str
    model: str


class LLMGenerator:
    """Generate context-grounded answers with a real provider or mock mode."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        mock_mode: bool | None = None,
    ):
        """Create a generator from explicit arguments or environment variables."""

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

    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> GenerationResult:
        """Generate an answer for a question using retrieved contexts."""

        context_list = list(contexts)
        prompt = build_prompt(question, context_list)

        if self.mock_mode:
            answer = self._generate_mock_answer(context_list)
        else:
            answer = self._generate_with_provider(prompt)

        return GenerationResult(
            question=question,
            answer=answer,
            contexts=context_list,
            prompt=prompt,
            model=self.model,
        )

    def _generate_mock_answer(self, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return "The provided context is insufficient."

        first_context = contexts[0].text.strip()
        first_sentence = _first_sentence(first_context)
        if not first_sentence:
            return "The provided context is insufficient."
        return f"Based on the provided context, {first_sentence} [1]"

    def _generate_with_provider(self, prompt: str) -> str:
        provider_config = PROVIDER_CONFIG.get(self.provider)
        if not provider_config:
            supported = ", ".join(sorted(PROVIDER_CONFIG))
            return f"Unsupported LLM provider '{self.provider}'. Supported providers: {supported}. Use MOCK_LLM=true for local testing."

        api_key_env = provider_config["api_key_env"]
        api_key = os.getenv(api_key_env)
        if not api_key or api_key == "your_key":
            return f"LLM API key is missing for provider '{self.provider}'. Set {api_key_env} or enable MOCK_LLM=true."

        try:
            from openai import OpenAI
        except ImportError:
            return "The openai package is not installed. Install dependencies with `pip install -r requirements.txt` or enable MOCK_LLM=true."

        client_kwargs = {"api_key": api_key}
        if provider_config["base_url"]:
            client_kwargs["base_url"] = provider_config["base_url"]

        try:
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
        except Exception as exc:
            return f"LLM generation failed for provider '{self.provider}': {exc}"

        content = response.choices[0].message.content if response.choices else None
        return content.strip() if content else "The provided context is insufficient."


def build_prompt(question: str, contexts: Sequence[RetrievedChunk]) -> str:
    """Build the strict context-grounded answer prompt."""

    context_block = _format_contexts(contexts)
    return PROMPT_TEMPLATE.format(question=question.strip(), context=context_block)


def _format_contexts(contexts: Sequence[RetrievedChunk]) -> str:
    if not contexts:
        return "[1] No retrieved context was provided."
    return "\n\n".join(f"[{index}] {chunk.text.strip()}" for index, chunk in enumerate(contexts, start=1))


def _first_sentence(text: str) -> str:
    for separator in [". ", "? ", "! ", "\n"]:
        if separator in text:
            return text.split(separator, 1)[0].strip().rstrip(".?!") + "."
    return text.strip()


def _env_bool(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}
