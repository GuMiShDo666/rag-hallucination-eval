"""Run the local RAG hallucination test.

The experiment uses an existing RAGBench-style eval file with `gold_context`,
`candidate_answer`, and `risk_type` fields. It builds a local FAISS retriever
from the gold contexts, then evaluates hallucination detection in two modes:

- oracle_context: judge the answer against its gold context
- retrieved_context: retrieve contexts from the local corpus first, then judge
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunker import split_documents
from src.document_loader import Document
from src.embedder import Embedder
from src.evaluator import RAGEvaluator
from src.retriever import FaissRetriever, RetrievedChunk


SUPPORTED_LABELS = {"supported"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the local RAG hallucination test.")
    parser.add_argument("--eval", default="data/eval_sets/ragbench_covidqa_1000.json")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="results/test")
    parser.add_argument("--report", default="docs/test.md")
    return parser.parse_args()


def main() -> None:
    """Run the experiment and write CSV, JSON, and Markdown outputs."""

    args = parse_args()
    items = load_eval_items(args.eval, args.limit)
    retriever = build_local_retriever(
        items=items,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    rows = evaluate_items(items, retriever, top_k=args.top_k)
    summary = build_summary(rows, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_sample.csv"
    summary_path = output_dir / "summary.json"
    write_csv(rows, csv_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary, csv_path, summary_path), encoding="utf-8")

    print(f"Evaluated {len(rows)} rows")
    print(f"oracle_context:   {format_metrics(summary['oracle_context'])}")
    print(f"retrieved_context: {format_metrics(summary['retrieved_context'])}")
    print(f"retrieval_source_hit_rate: {summary['retrieval']['source_hit_rate']:.4f}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


def load_eval_items(eval_path: str, limit: int) -> list[dict[str, Any]]:
    """Load eval rows with fields required for the local test."""

    path = Path(eval_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Evaluation set must be a non-empty JSON list.")

    required = {"question", "gold_context", "candidate_answer", "risk_type"}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload[:limit]):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation item {index} must be a JSON object.")
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"Evaluation item {index} is missing required fields: {missing}")
        rows.append(item)
    return rows


def build_local_retriever(items: list[dict[str, Any]], chunk_size: int, chunk_overlap: int) -> FaissRetriever:
    """Build a local RAG retriever from the gold contexts."""

    documents: list[Document] = []
    for index, item in enumerate(items):
        source_id = str(item.get("source_id") or index)
        documents.append(
            Document(
                text=str(item["gold_context"]),
                source=f"ragbench_{source_id}.md",
                page=None,
                metadata={
                    "source_id": source_id,
                    "source_name": item.get("source_name", "ragbench"),
                    "risk_type": item.get("risk_type", ""),
                },
            )
        )

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever = FaissRetriever(Embedder("hashing"))
    retriever.build_index(chunks)
    return retriever


def evaluate_items(items: list[dict[str, Any]], retriever: FaissRetriever, top_k: int) -> list[dict[str, Any]]:
    """Evaluate each sample against oracle context and retrieved context."""

    evaluator = RAGEvaluator()
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        question = str(item["question"])
        answer = str(item["candidate_answer"])
        gold_context = str(item["gold_context"])
        reference_answer = str(item.get("reference_answer", ""))
        risk_type = str(item["risk_type"])
        source_id = str(item.get("source_id") or index)

        retrieved = retriever.retrieve(question, top_k=top_k)
        retrieved_texts = [chunk.text for chunk in retrieved]

        oracle_eval = evaluator.evaluate_single(
            question=question,
            answer=answer,
            contexts=[gold_context],
            reference_answer=reference_answer,
            gold_context=gold_context,
        )
        retrieved_eval = evaluator.evaluate_single(
            question=question,
            answer=answer,
            contexts=retrieved_texts,
            reference_answer=reference_answer,
            gold_context=gold_context,
        )

        expected_hallucinated = expected_is_hallucinated(risk_type)
        rows.append(
            {
                "row_index": index,
                "source_id": source_id,
                "risk_type": risk_type,
                "expected_hallucinated": expected_hallucinated,
                "oracle_pred_hallucinated": prediction_is_hallucinated(
                    oracle_eval.raw["final_judgement"],
                    oracle_eval.hallucination_rate,
                ),
                "retrieved_pred_hallucinated": prediction_is_hallucinated(
                    retrieved_eval.raw["final_judgement"],
                    retrieved_eval.hallucination_rate,
                ),
                "oracle_final_judgement": oracle_eval.raw["final_judgement"],
                "retrieved_final_judgement": retrieved_eval.raw["final_judgement"],
                "oracle_hallucination_rate": oracle_eval.hallucination_rate,
                "retrieved_hallucination_rate": retrieved_eval.hallucination_rate,
                "oracle_faithfulness": oracle_eval.faithfulness,
                "retrieved_faithfulness": retrieved_eval.faithfulness,
                "retrieved_context_precision": retrieved_eval.context_precision,
                "retrieval_source_hit": retrieval_source_hit(retrieved, source_id),
                "retrieved_source_ids": "|".join(retrieved_source_ids(retrieved)),
                "max_retrieved_score": max((chunk.score for chunk in retrieved), default=0.0),
                "question": question,
                "answer": answer,
                "oracle_unsupported_spans": json.dumps(
                    oracle_eval.raw["unsupported_spans"],
                    ensure_ascii=False,
                ),
                "retrieved_unsupported_spans": json.dumps(
                    retrieved_eval.raw["unsupported_spans"],
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def expected_is_hallucinated(risk_type: str) -> bool:
    """Map dataset risk labels to a binary hallucination target."""

    return risk_type.strip().lower() not in SUPPORTED_LABELS


def prediction_is_hallucinated(final_judgement: str, hallucination_rate: float | None) -> bool:
    """Map detector output to a binary hallucination prediction."""

    if hallucination_rate is not None and hallucination_rate > 0:
        return True
    return final_judgement.strip().lower() != "supported"


def retrieved_source_ids(retrieved: list[RetrievedChunk]) -> list[str]:
    """Return source ids from retrieved chunks."""

    ids: list[str] = []
    for chunk in retrieved:
        source_id = chunk.metadata.get("source_id")
        if source_id is not None:
            ids.append(str(source_id))
    return ids


def retrieval_source_hit(retrieved: list[RetrievedChunk], expected_source_id: str) -> bool:
    """Whether retrieval returned at least one chunk from the gold context document."""

    return expected_source_id in retrieved_source_ids(retrieved)


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write per-sample rows without requiring pandas."""

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Build aggregate metrics for the experiment."""

    risk_types = [str(row["risk_type"]) for row in rows]
    return {
        "dataset": args.eval,
        "num_samples": int(len(rows)),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
        "label_distribution": {label: risk_types.count(label) for label in sorted(set(risk_types))},
        "oracle_context": classification_metrics(
            [bool(row["expected_hallucinated"]) for row in rows],
            [bool(row["oracle_pred_hallucinated"]) for row in rows],
        ),
        "retrieved_context": classification_metrics(
            [bool(row["expected_hallucinated"]) for row in rows],
            [bool(row["retrieved_pred_hallucinated"]) for row in rows],
        ),
        "averages": {
            "oracle_hallucination_rate": _mean(rows, "oracle_hallucination_rate"),
            "retrieved_hallucination_rate": _mean(rows, "retrieved_hallucination_rate"),
            "oracle_faithfulness": _mean(rows, "oracle_faithfulness"),
            "retrieved_faithfulness": _mean(rows, "retrieved_faithfulness"),
            "retrieved_context_precision": _mean(rows, "retrieved_context_precision"),
        },
        "retrieval": {
            "source_hit_rate": _mean(rows, "retrieval_source_hit"),
            "avg_max_retrieved_score": _mean(rows, "max_retrieved_score"),
        },
    }


def classification_metrics(expected: list[bool], predicted: list[bool]) -> dict[str, float | int]:
    """Compute binary classification metrics for hallucination detection."""

    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must have the same length.")

    tp = sum(bool(e) and bool(p) for e, p in zip(expected, predicted, strict=True))
    fp = sum((not bool(e)) and bool(p) for e, p in zip(expected, predicted, strict=True))
    tn = sum((not bool(e)) and (not bool(p)) for e, p in zip(expected, predicted, strict=True))
    fn = sum(bool(e) and (not bool(p)) for e, p in zip(expected, predicted, strict=True))
    total = len(expected)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": _safe_div(tp + fp, total),
        "actual_positive_rate": _safe_div(tp + fn, total),
    }


def render_report(summary: dict[str, Any], csv_path: Path, summary_path: Path) -> str:
    """Render a Markdown report."""

    oracle = summary["oracle_context"]
    retrieved = summary["retrieved_context"]
    averages = summary["averages"]
    retrieval = summary["retrieval"]

    return f"""# Local RAG Test

This experiment builds a local RAG corpus from `{summary['dataset']}` and evaluates RAGBench candidate answers against the project hallucination detector.

## Setup

| Setting | Value |
|---|---:|
| Samples | {summary['num_samples']} |
| Chunk size | {summary['chunk_size']} |
| Chunk overlap | {summary['chunk_overlap']} |
| Top-k | {summary['top_k']} |
| Supported rows | {summary['label_distribution'].get('supported', 0)} |
| Unsupported rows | {summary['label_distribution'].get('unsupported', 0)} |

Two modes were measured:

- `oracle_context`: detect `candidate_answer` against the gold context.
- `retrieved_context`: retrieve contexts from the local FAISS corpus first, then detect.

## Hallucination Detection Metrics

| Mode | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_context | {oracle['accuracy']:.4f} | {oracle['precision']:.4f} | {oracle['recall']:.4f} | {oracle['f1']:.4f} | {oracle['tp']} | {oracle['fp']} | {oracle['tn']} | {oracle['fn']} |
| retrieved_context | {retrieved['accuracy']:.4f} | {retrieved['precision']:.4f} | {retrieved['recall']:.4f} | {retrieved['f1']:.4f} | {retrieved['tp']} | {retrieved['fp']} | {retrieved['tn']} | {retrieved['fn']} |

## Retrieval Metrics

| Metric | Value |
|---|---:|
| source_hit_rate | {retrieval['source_hit_rate']:.4f} |
| avg_max_retrieved_score | {retrieval['avg_max_retrieved_score']:.4f} |
| retrieved_context_precision | {averages['retrieved_context_precision']:.4f} |

## Average Scores

| Metric | Value |
|---|---:|
| oracle_hallucination_rate | {averages['oracle_hallucination_rate']:.4f} |
| retrieved_hallucination_rate | {averages['retrieved_hallucination_rate']:.4f} |
| oracle_faithfulness | {averages['oracle_faithfulness']:.4f} |
| retrieved_faithfulness | {averages['retrieved_faithfulness']:.4f} |

## Interpretation

The previous 100% results came from the tiny 5-row sample set and mock generation that copied heavily from retrieved context. On the RAGBench sample, the current local fallback detector has high overall accuracy mainly because supported answers dominate the dataset, but it has low precision and recall for hallucinated or unsupported answers.

The main bottleneck is the detector: it relies on lexical overlap, so it misses many unsupported answers that reuse terms from the context, and it can flag supported answers when wording differs from the gold context.

## Output Files

| File | Description |
|---|---|
| `{csv_path}` | Per-sample predictions and metrics |
| `{summary_path}` | Aggregate metrics JSON |
"""


def format_metrics(metrics: dict[str, float | int]) -> str:
    """Format metric summary for terminal output."""

    return (
        f"accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(rows: list[dict[str, Any]], column: str) -> float:
    values = [float(row[column]) for row in rows if row.get(column) is not None]
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
