"""Hallucination detection for RAG answers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Sequence

from src.retriever import RetrievedChunk


ALLOWED_STATUSES = {"supported", "unsupported", "contradicted", "unclear"}
ALLOWED_JUDGEMENTS = {"supported", "partially_supported", "unsupported"}

JUDGE_PROMPT_TEMPLATE = """You are a factual consistency judge for a RAG system.

Given a question, retrieved context, and an answer, split the answer into atomic claims.
For each claim, judge whether it is supported by the context.

Labels:
- supported: the context clearly supports the claim
- unsupported: the context does not provide enough evidence
- contradicted: the context contradicts the claim
- unclear: the claim is ambiguous or cannot be judged

Return valid JSON only.

Question:
{question}

Context:
{contexts}

Answer:
{answer}

Return format:
{{
  "claims": [
    {{
      "text": "...",
      "status": "supported",
      "reason": "..."
    }}
  ],
  "final_judgement": "supported"
}}"""


@dataclass
class HallucinationSpan:
    """A judged answer claim or span."""

    text: str
    status: str
    reason: str
    confidence: float | None = None


@dataclass
class HallucinationResult:
    """Hallucination detection result for one answer."""

    spans: list[HallucinationSpan]
    hallucination_rate: float
    final_judgement: str


class HallucinationDetector:
    """Detect unsupported answer claims with optional LLM judge and local fallback."""

    def __init__(
        self,
        provider: str = "qwen",
        model: str | None = None,
        use_llm_judge: bool = False,
    ):
        """Create a detector.

        `use_llm_judge` is off by default so local tests do not call an API.
        When enabled, Qwen is the default provider.
        """

        self.provider = provider.lower()
        self.model = model or os.getenv("QWEN_MODEL", "qwen-plus")
        self.use_llm_judge = use_llm_judge

    def detect(
        self,
        question: str,
        contexts: Sequence[str | RetrievedChunk],
        answer: str,
    ) -> HallucinationResult:
        """Detect unsupported, contradicted, unclear, and supported claims."""

        context_texts = _context_texts(contexts)
        if not answer.strip():
            return HallucinationResult(
                spans=[
                    HallucinationSpan(
                        text="",
                        status="unclear",
                        reason="Answer is empty.",
                    )
                ],
                hallucination_rate=0.0,
                final_judgement="unsupported",
            )

        if self.use_llm_judge:
            judged = self._detect_with_llm(question, context_texts, answer)
            if judged is not None:
                return judged

        return self._detect_with_local_fallback(context_texts, answer)

    def build_judge_prompt(self, question: str, contexts: Sequence[str], answer: str) -> str:
        """Build the LLM-as-a-Judge prompt without calling an API."""

        return JUDGE_PROMPT_TEMPLATE.format(
            question=question.strip(),
            contexts="\n\n".join(contexts).strip(),
            answer=answer.strip(),
        )

    def _detect_with_llm(
        self,
        question: str,
        contexts: Sequence[str],
        answer: str,
    ) -> HallucinationResult | None:
        api_key = os.getenv("QWEN_API_KEY")
        if self.provider != "qwen" or not api_key or api_key == "your_key":
            return None

        try:
            from openai import OpenAI
        except ImportError:
            return None

        prompt = self.build_judge_prompt(question, contexts, answer)
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = response.choices[0].message.content if response.choices else ""
            return parse_judge_response(content)
        except Exception:
            return None

    def _detect_with_local_fallback(self, contexts: Sequence[str], answer: str) -> HallucinationResult:
        claims = split_into_claims(answer)
        if not claims:
            return HallucinationResult(
                spans=[
                    HallucinationSpan(
                        text=answer,
                        status="unclear",
                        reason="Could not split the answer into claims.",
                    )
                ],
                hallucination_rate=0.0,
                final_judgement="unsupported",
            )

        context_text = " ".join(contexts)
        spans = [_judge_claim_locally(claim, context_text) for claim in claims]
        return _build_result(spans)


def parse_judge_response(raw: str) -> HallucinationResult:
    """Parse an LLM judge JSON response with robust extraction fallback."""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = _extract_json_object(raw)

    if not isinstance(payload, dict):
        return _unclear_result(raw, "Judge response was not valid JSON.")

    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return _unclear_result(raw, "Judge response did not contain a non-empty claims list.")

    spans: list[HallucinationSpan] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("text", "")).strip()
        status = str(claim.get("status", "unclear")).strip().lower()
        reason = str(claim.get("reason", "")).strip() or "No reason provided."
        confidence = claim.get("confidence")
        if status not in ALLOWED_STATUSES:
            status = "unclear"
        if not text:
            continue
        spans.append(
            HallucinationSpan(
                text=text,
                status=status,
                reason=reason,
                confidence=confidence if isinstance(confidence, int | float) else None,
            )
        )

    if not spans:
        return _unclear_result(raw, "Judge response claims were empty after parsing.")

    judgement = str(payload.get("final_judgement", "")).strip().lower()
    result = _build_result(spans)
    if judgement in ALLOWED_JUDGEMENTS:
        result.final_judgement = judgement
    return result


def split_into_claims(answer: str) -> list[str]:
    """Split an answer into simple sentence-level claims."""

    cleaned = re.sub(r"\s+", " ", answer.strip())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    claims: list[str] = []
    for part in parts:
        claim = part.strip()
        if not claim:
            continue
        if re.fullmatch(r"(?:\[\d+\]\s*)+", claim):
            if claims:
                claims[-1] = f"{claims[-1]} {claim}".strip()
            continue
        claims.append(claim)
    return claims


def _context_texts(contexts: Sequence[str | RetrievedChunk]) -> list[str]:
    texts: list[str] = []
    for context in contexts:
        if isinstance(context, RetrievedChunk):
            texts.append(context.text)
        else:
            texts.append(str(context))
    return [text for text in texts if text.strip()]


def _judge_claim_locally(claim: str, context_text: str) -> HallucinationSpan:
    if not context_text.strip():
        return HallucinationSpan(
            text=claim,
            status="unsupported",
            reason="No retrieved context was provided.",
            confidence=0.8,
        )

    claim_tokens = _content_tokens(claim)
    context_tokens = _content_tokens(context_text)
    if not claim_tokens:
        return HallucinationSpan(
            text=claim,
            status="unclear",
            reason="Claim has too little content to judge.",
            confidence=0.5,
        )

    overlap = len(claim_tokens & context_tokens) / len(claim_tokens)
    if overlap >= 0.55:
        return HallucinationSpan(
            text=claim,
            status="supported",
            reason="Most claim terms appear in the retrieved context.",
            confidence=round(overlap, 3),
        )
    return HallucinationSpan(
        text=claim,
        status="unsupported",
        reason="The retrieved context does not provide enough lexical evidence for the claim.",
        confidence=round(1 - overlap, 3),
    )


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "based",
        "be",
        "by",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())
    return {token for token in tokens if token not in stopwords and len(token) > 1}


def _build_result(spans: list[HallucinationSpan]) -> HallucinationResult:
    unsupported_count = sum(span.status in {"unsupported", "contradicted"} for span in spans)
    rate = unsupported_count / len(spans) if spans else 0.0
    supported_count = sum(span.status == "supported" for span in spans)

    if spans and supported_count == len(spans):
        judgement = "supported"
    elif supported_count > 0:
        judgement = "partially_supported"
    else:
        judgement = "unsupported"

    return HallucinationResult(
        spans=spans,
        hallucination_rate=rate,
        final_judgement=judgement,
    )


def _extract_json_object(raw: str) -> Any:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _unclear_result(text: str, reason: str) -> HallucinationResult:
    span_text = text.strip() or "unparseable judge response"
    return HallucinationResult(
        spans=[
            HallucinationSpan(
                text=span_text,
                status="unclear",
                reason=reason,
            )
        ],
        hallucination_rate=0.0,
        final_judgement="unsupported",
    )
