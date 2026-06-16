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

    id: str | None = None
    reference_answer: str | None = None
    gold_context: str | None = None


class BatchEvaluateRequest(BaseModel):
    """Request body for batch external RAG evaluation."""

    items: list[EvaluateRequest] = Field(..., min_length=1)


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

    id: str | None = None
    question: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    citation_accuracy: float | None


class BatchSummary(BaseModel):
    """Aggregate metrics for a batch evaluation response."""

    total: int
    supported: int
    partially_supported: int
    unsupported: int
    avg_hallucination_rate: float | None
    avg_faithfulness: float | None
    avg_answer_relevancy: float | None
    avg_context_precision: float | None
    avg_citation_accuracy: float | None


class BatchEvaluateResponse(BaseModel):
    """Batch evaluation response."""

    summary: BatchSummary
    results: list[EvaluateResponse]


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

    return _evaluate_item(request)


@app.post("/batch_evaluate", response_model=BatchEvaluateResponse)
def batch_evaluate(request: BatchEvaluateRequest) -> BatchEvaluateResponse:
    """Evaluate multiple external RAG outputs in one request."""

    results = [_evaluate_item(item) for item in request.items]
    return BatchEvaluateResponse(
        summary=_batch_summary(results),
        results=results,
    )


def _evaluate_item(request: EvaluateRequest) -> EvaluateResponse:
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
        id=request.id,
        question=request.question,
        faithfulness=result.faithfulness,
        answer_relevancy=result.answer_relevancy,
        context_precision=result.context_precision,
        citation_accuracy=result.citation_accuracy,
    )


def _batch_summary(results: list[EvaluateResponse]) -> BatchSummary:
    judgements = [result.final_judgement for result in results]
    return BatchSummary(
        total=len(results),
        supported=judgements.count("supported"),
        partially_supported=judgements.count("partially_supported"),
        unsupported=judgements.count("unsupported"),
        avg_hallucination_rate=_mean([result.hallucination_rate for result in results]),
        avg_faithfulness=_mean([result.faithfulness for result in results]),
        avg_answer_relevancy=_mean([result.answer_relevancy for result in results]),
        avg_context_precision=_mean([result.context_precision for result in results]),
        avg_citation_accuracy=_mean([result.citation_accuracy for result in results]),
    )


def _mean(values: list[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


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
