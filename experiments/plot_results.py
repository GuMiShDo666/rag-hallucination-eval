"""Generate plots from ablation experiment results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse plot generation arguments."""

    parser = argparse.ArgumentParser(description="Plot RAG hallucination evaluation results.")
    parser.add_argument("--input", default="results/ablation_results.csv", help="Ablation CSV path.")
    parser.add_argument("--output_dir", default="results/figures", help="Directory for PNG figures.")
    return parser.parse_args()


def main() -> None:
    """Read ablation results and save all required figures."""

    args = parse_args()
    dataframe = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = [
        _plot_chunk_size_hallucination(dataframe, output_dir),
        _plot_top_k_faithfulness(dataframe, output_dir),
        _plot_reranker_comparison(dataframe, output_dir),
        _plot_query_rewrite_comparison(dataframe, output_dir),
    ]
    for path in saved_paths:
        print(f"Saved {path}")


def _plot_chunk_size_hallucination(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    subset = dataframe[dataframe["setting_name"].str.startswith("chunk_size_")].copy()
    subset = subset.sort_values("chunk_size")
    path = output_dir / "chunk_size_hallucination_rate.png"
    _bar_plot(
        labels=[str(value) for value in subset["chunk_size"]],
        values=subset["avg_hallucination_rate"],
        title="Hallucination Rate by Chunk Size",
        xlabel="Chunk Size",
        ylabel="Average Hallucination Rate",
        output_path=path,
    )
    return path


def _plot_top_k_faithfulness(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    subset = dataframe[dataframe["setting_name"].str.startswith("top_k_")].copy()
    subset = subset.sort_values("top_k")
    path = output_dir / "top_k_faithfulness.png"
    _bar_plot(
        labels=[str(value) for value in subset["top_k"]],
        values=subset["avg_faithfulness"],
        title="Faithfulness by Top-K",
        xlabel="Top-K Retrieved Chunks",
        ylabel="Average Faithfulness",
        output_path=path,
    )
    return path


def _plot_reranker_comparison(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    subset = dataframe[dataframe["setting_name"].str.startswith("reranker_")].copy()
    subset = subset.sort_values("use_reranker")
    path = output_dir / "baseline_vs_reranker.png"
    _grouped_metric_plot(
        labels=["Baseline", "Reranker"],
        values=[
            _first_metric(subset, "use_reranker", False, "avg_faithfulness"),
            _first_metric(subset, "use_reranker", True, "avg_faithfulness"),
        ],
        title="Baseline vs Reranker Faithfulness",
        ylabel="Average Faithfulness",
        output_path=path,
    )
    return path


def _plot_query_rewrite_comparison(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    subset = dataframe[dataframe["setting_name"].str.startswith("query_rewrite_")].copy()
    subset = subset.sort_values("use_query_rewrite")
    path = output_dir / "baseline_vs_query_rewrite.png"
    _grouped_metric_plot(
        labels=["Baseline", "Query Rewrite"],
        values=[
            _first_metric(subset, "use_query_rewrite", False, "avg_faithfulness"),
            _first_metric(subset, "use_query_rewrite", True, "avg_faithfulness"),
        ],
        title="Baseline vs Query Rewrite Faithfulness",
        ylabel="Average Faithfulness",
        output_path=path,
    )
    return path


def _bar_plot(labels: list[str], values, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    numeric_values = [0.0 if pd.isna(value) else float(value) for value in values]
    ax.bar(labels, numeric_values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, max(numeric_values, default=0.0) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _grouped_metric_plot(labels: list[str], values: list[float | None], title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    numeric_values = [0.0 if value is None or pd.isna(value) else float(value) for value in values]
    ax.bar(labels, numeric_values, color=["#4C78A8", "#F58518"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, max(numeric_values, default=0.0) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _first_metric(dataframe: pd.DataFrame, column: str, expected_value: bool, metric: str) -> float | None:
    subset = dataframe[dataframe[column] == expected_value]
    if subset.empty:
        return None
    value = subset.iloc[0][metric]
    return None if pd.isna(value) else float(value)


if __name__ == "__main__":
    main()
