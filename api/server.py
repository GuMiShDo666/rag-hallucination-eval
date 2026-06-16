"""FastAPI server exposing hallucination detection and evaluation endpoints."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.evaluator import RAGEvaluator
from src.hallucination_detector import HallucinationDetector


class DetectRequest(BaseModel):
    """Request body for claim-level hallucination detection."""

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    contexts: list[str] = Field(default_factory=list)
    use_llm_judge: bool = False


class EvaluateRequest(DetectRequest):
    """Request body for detection plus metric calculation."""

    reference_answer: str | None = None
    gold_context: str | None = None


class SpanResponse(BaseModel):
    """One judged answer span."""

    text: str
    status: Literal["supported", "unsupported", "contradicted", "unclear"]
    reason: str
    confidence: float | None = None


class DetectResponse(BaseModel):
    """Claim-level hallucination detection response."""

    spans: list[SpanResponse]
    unsupported_spans: list[SpanResponse]
    hallucination_rate: float
    final_judgement: str


class EvaluateResponse(DetectResponse):
    """Detection response with fallback evaluation metrics."""

    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    citation_accuracy: float | None


app = FastAPI(
    title="RAG Hallucination Eval API",
    version="0.1.0",
    description="HTTP API for detecting unsupported claims in RAG answers.",
)


@app.get("/health")
def health() -> dict:
    """Return API health status."""

    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    """Detect unsupported or contradicted spans in a generated answer."""

    detector = _detector(use_llm_judge=request.use_llm_judge)
    result = detector.detect(
        question=request.question,
        contexts=request.contexts,
        answer=request.answer,
    )
    return _detect_response(result)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Detect hallucinations and compute fallback evaluation metrics."""

    detector = _detector(use_llm_judge=request.use_llm_judge)
    evaluator = RAGEvaluator(detector=detector)
    result = evaluator.evaluate_single(
        question=request.question,
        answer=request.answer,
        contexts=request.contexts,
        reference_answer=request.reference_answer,
        gold_context=request.gold_context,
    )
    detection = detector.detect(
        question=request.question,
        contexts=request.contexts,
        answer=request.answer,
    )
    response = _detect_response(detection)
    return EvaluateResponse(
        **response.model_dump(),
        faithfulness=result.faithfulness,
        answer_relevancy=result.answer_relevancy,
        context_precision=result.context_precision,
        citation_accuracy=result.citation_accuracy,
    )


@lru_cache(maxsize=2)
def _detector(use_llm_judge: bool) -> HallucinationDetector:
    return HallucinationDetector(use_llm_judge=use_llm_judge)


def _detect_response(result) -> DetectResponse:
    spans = [
        SpanResponse(
            text=span.text,
            status=span.status,
            reason=span.reason,
            confidence=span.confidence,
        )
        for span in result.spans
    ]
    unsupported = [span for span in spans if span.status in {"unsupported", "contradicted"}]
    return DetectResponse(
        spans=spans,
        unsupported_spans=unsupported,
        hallucination_rate=result.hallucination_rate,
        final_judgement=result.final_judgement,
    )
