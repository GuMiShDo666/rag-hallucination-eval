"""Run ablation experiments for RAG hallucination evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from src.embedder import Embedder
from src.evaluator import RAGEvaluator
from src.pipeline import RAGHallucinationPipeline


def parse_args() -> argparse.Namespace:
    """Parse ablation experiment arguments."""

    parser = argparse.ArgumentParser(description="Run RAG retrieval-strategy ablation experiments.")
    parser.add_argument("--docs", default="data/raw_docs", help="Directory containing source documents.")
    parser.add_argument("--eval", default="data/eval_set.json", help="Evaluation set JSON path.")
    parser.add_argument("--output", default="results/ablation_results.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    """Run every ablation setting and save aggregate metrics."""

    args = parse_args()
    settings = _build_settings()
    rows: list[dict] = []

    for index, setting in enumerate(settings, start=1):
        print(f"[{index}/{len(settings)}] Running {setting['setting_name']}")
        try:
            row = _run_setting(args.docs, args.eval, setting)
            print(
                "  ok "
                f"faithfulness={_fmt(row['avg_faithfulness'])} "
                f"hallucination_rate={_fmt(row['avg_hallucination_rate'])}"
            )
        except Exception as exc:
            row = {
                **setting,
                "avg_faithfulness": None,
                "avg_answer_relevancy": None,
                "avg_context_precision": None,
                "avg_hallucination_rate": None,
                "num_samples": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  failed: {row['error']}")
            traceback.print_exc(limit=1)
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)
    print(f"Saved {len(dataframe)} ablation rows to {args.output}")


def _build_settings() -> list[dict]:
    settings: list[dict] = []
    for chunk_size in [256, 512, 1024]:
        settings.append(
            {
                "setting_name": f"chunk_size_{chunk_size}",
                "chunk_size": chunk_size,
                "top_k": 5,
                "use_query_rewrite": False,
                "use_reranker": False,
            }
        )
    for top_k in [3, 5, 8]:
        settings.append(
            {
                "setting_name": f"top_k_{top_k}",
                "chunk_size": 512,
                "top_k": top_k,
                "use_query_rewrite": False,
                "use_reranker": False,
            }
        )
    for use_query_rewrite in [False, True]:
        settings.append(
            {
                "setting_name": f"query_rewrite_{str(use_query_rewrite).lower()}",
                "chunk_size": 512,
                "top_k": 5,
                "use_query_rewrite": use_query_rewrite,
                "use_reranker": False,
            }
        )
    for use_reranker in [False, True]:
        settings.append(
            {
                "setting_name": f"reranker_{str(use_reranker).lower()}",
                "chunk_size": 512,
                "top_k": 5,
                "use_query_rewrite": False,
                "use_reranker": use_reranker,
            }
        )
    return settings


def _run_setting(docs_path: str, eval_path: str, setting: dict) -> dict:
    pipeline = RAGHallucinationPipeline(
        docs_path=docs_path,
        chunk_size=setting["chunk_size"],
        chunk_overlap=min(80, max(0, setting["chunk_size"] // 4)),
        top_k=setting["top_k"],
        use_query_rewrite=setting["use_query_rewrite"],
        use_reranker=setting["use_reranker"],
        embedder=Embedder("hashing"),
    )
    pipeline.build()

    evaluator = RAGEvaluator()
    per_sample_output = Path("results") / "ablation_runs" / f"{setting['setting_name']}.csv"
    dataframe = evaluator.evaluate_batch(eval_path, pipeline, str(per_sample_output))

    return {
        **setting,
        "avg_faithfulness": _mean(dataframe, "faithfulness"),
        "avg_answer_relevancy": _mean(dataframe, "answer_relevancy"),
        "avg_context_precision": _mean(dataframe, "context_precision"),
        "avg_hallucination_rate": _mean(dataframe, "hallucination_rate"),
        "num_samples": len(dataframe),
        "error": "",
    }


def _mean(dataframe: pd.DataFrame, column: str) -> float | None:
    if column not in dataframe:
        return None
    value = dataframe[column].dropna().mean()
    return float(value) if value == value else None


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
