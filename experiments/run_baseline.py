"""Run the baseline RAG hallucination evaluation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.embedder import Embedder
from src.evaluator import RAGEvaluator
from src.pipeline import RAGHallucinationPipeline


def parse_args() -> argparse.Namespace:
    """Parse baseline experiment arguments."""

    parser = argparse.ArgumentParser(description="Run baseline RAG hallucination evaluation.")
    parser.add_argument("--docs", default="data/documents", help="Directory containing source documents.")
    parser.add_argument("--eval", default="data/eval_sets/sample.json", help="Evaluation set JSON path.")
    parser.add_argument("--output", default="results/baseline_results.csv", help="Output CSV path.")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=80)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Build the baseline pipeline, run evaluation, and save CSV results."""

    args = parse_args()
    _ensure_inputs(args.docs, args.eval)

    pipeline = RAGHallucinationPipeline(
        docs_path=args.docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        embedder=Embedder("hashing"),
    )
    pipeline.build()

    evaluator = RAGEvaluator()
    dataframe = evaluator.evaluate_batch(args.eval, pipeline, args.output)
    print(f"Saved {len(dataframe)} rows to {args.output}")
    _print_average_metrics(dataframe)


def _ensure_inputs(docs_path: str, eval_path: str) -> None:
    docs_dir = Path(docs_path)
    eval_file = Path(eval_path)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {docs_path}")
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation set does not exist: {eval_path}")

    with eval_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Evaluation set must be a non-empty JSON list.")


def _print_average_metrics(dataframe) -> None:
    metric_columns = {
        "faithfulness": "avg_faithfulness",
        "answer_relevancy": "avg_answer_relevancy",
        "context_precision": "avg_context_precision",
        "hallucination_rate": "avg_hallucination_rate",
    }
    for column, label in metric_columns.items():
        if column not in dataframe:
            print(f"{label}: None")
            continue
        value = dataframe[column].dropna().mean()
        print(f"{label}: {value:.4f}" if value == value else f"{label}: None")


if __name__ == "__main__":
    main()
