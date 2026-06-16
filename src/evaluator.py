"""Automatic RAG evaluation with stable fallback metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.hallucination_detector import HallucinationDetector


@dataclass
class EvaluationResult:
    """Evaluation metrics for one RAG answer."""

    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    hallucination_rate: float | None
    citation_accuracy: float | None
    raw: dict


class RAGEvaluator:
    """Evaluate RAG outputs with fallback metrics and optional future integrations."""

    def __init__(self, generator: Any | None = None, detector: HallucinationDetector | None = None):
        """Create an evaluator.

        `generator` is accepted for later LLM-as-a-Judge integrations. The
        default implementation is deterministic and does not call an API.
        """

        self.generator = generator
        self.detector = detector or HallucinationDetector(use_llm_judge=False)

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference_answer: str | None = None,
        gold_context: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a single generated answer against retrieved contexts."""

        hallucination = self.detector.detect(question, contexts, answer)
        faithfulness = _faithfulness_from_hallucination_rate(hallucination.hallucination_rate)
        answer_relevancy = _lexical_overlap_score(answer, reference_answer or question)
        context_precision = _context_precision(contexts, gold_context)
        citation_accuracy = _citation_accuracy(answer, len(contexts))

        raw = {
            "final_judgement": hallucination.final_judgement,
            "unsupported_spans": [
                span.text
                for span in hallucination.spans
                if span.status in {"unsupported", "contradicted"}
            ],
            "spans": [span.__dict__ for span in hallucination.spans],
        }

        return EvaluationResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            hallucination_rate=hallucination.hallucination_rate,
            citation_accuracy=citation_accuracy,
            raw=raw,
        )

    def evaluate_batch(self, eval_set_path: str, pipeline: Any, output_csv_path: str) -> pd.DataFrame:
        """Run a pipeline over an eval set and save one row per question."""

        eval_items = _load_eval_set(eval_set_path)
        rows: list[dict] = []
        for item in eval_items:
            question = item["question"]
            pipeline_result = pipeline.ask(question)
            answer = str(pipeline_result.get("answer", ""))
            contexts = _extract_context_texts(pipeline_result.get("contexts", []))
            evaluation = self.evaluate_single(
                question=question,
                answer=answer,
                contexts=contexts,
                reference_answer=item.get("reference_answer"),
                gold_context=item.get("gold_context"),
            )

            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "faithfulness": evaluation.faithfulness,
                    "answer_relevancy": evaluation.answer_relevancy,
                    "context_precision": evaluation.context_precision,
                    "hallucination_rate": evaluation.hallucination_rate,
                    "citation_accuracy": evaluation.citation_accuracy,
                    "retrieved_chunk_ids": _join_chunk_ids(pipeline_result.get("contexts", [])),
                    "unsupported_spans": json.dumps(evaluation.raw["unsupported_spans"], ensure_ascii=False),
                }
            )

        dataframe = pd.DataFrame(rows)
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        return dataframe


def _load_eval_set(eval_set_path: str) -> list[dict]:
    path = Path(eval_set_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Evaluation set must be a JSON list.")
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not item.get("question"):
            raise ValueError(f"Evaluation item {index} must contain a question.")
    return payload


def _faithfulness_from_hallucination_rate(rate: float | None) -> float | None:
    if rate is None:
        return None
    return max(0.0, min(1.0, 1.0 - rate))


def _context_precision(contexts: list[str], gold_context: str | None) -> float | None:
    if not gold_context:
        return None
    if not contexts:
        return 0.0

    gold_tokens = _content_tokens(gold_context)
    if not gold_tokens:
        return None

    relevant_count = 0
    for context in contexts:
        context_tokens = _content_tokens(context)
        overlap = len(gold_tokens & context_tokens) / len(gold_tokens)
        if overlap >= 0.5 or gold_context.lower() in context.lower():
            relevant_count += 1
    return relevant_count / len(contexts)


def _citation_accuracy(answer: str, num_contexts: int) -> float | None:
    citations = [int(match) for match in re.findall(r"\[(\d+)\]", answer)]
    if not citations:
        return None
    valid = sum(1 for citation in citations if 1 <= citation <= num_contexts)
    return valid / len(citations)


def _lexical_overlap_score(text: str, target: str) -> float:
    text_tokens = _content_tokens(text)
    target_tokens = _content_tokens(target)
    if not target_tokens:
        return 0.0
    if not text_tokens:
        return 0.0
    return len(text_tokens & target_tokens) / len(target_tokens)


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
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


def _extract_context_texts(contexts: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(contexts, list):
        return texts
    for context in contexts:
        if isinstance(context, str):
            texts.append(context)
        elif isinstance(context, dict):
            texts.append(str(context.get("text", "")))
        elif hasattr(context, "text"):
            texts.append(str(context.text))
    return [text for text in texts if text.strip()]


def _join_chunk_ids(contexts: Any) -> str:
    chunk_ids: list[str] = []
    if not isinstance(contexts, list):
        return ""
    for context in contexts:
        if isinstance(context, dict):
            chunk_id = context.get("chunk_id")
        else:
            chunk_id = getattr(context, "chunk_id", None)
        if chunk_id:
            chunk_ids.append(str(chunk_id))
    return "|".join(chunk_ids)
