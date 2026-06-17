"""Plot the local RAGBench test summary for README and docs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/.cache").resolve()))
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


MODE_LABELS = {
    "oracle_context": "Oracle context",
    "retrieved_context": "Retrieved context",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Plot local RAGBench test results.")
    parser.add_argument("--summary", default="results/test/summary.json", help="Test summary JSON path.")
    parser.add_argument("--output", default="docs/assets/test_results", help="Output path without suffix.")
    return parser.parse_args()


def main() -> None:
    """Read the summary JSON and save README-ready result figures."""

    args = parse_args()
    summary = _load_summary(Path(args.summary))
    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    fig = _build_figure(summary)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_base.with_suffix('.png')}")
    print(f"Saved {output_base.with_suffix('.svg')}")


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_figure(summary: dict):
    _configure_matplotlib()
    fig = plt.figure(figsize=(13.4, 7.8), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.08],
        width_ratios=[1.2, 1.0, 1.0],
        left=0.06,
        right=0.985,
        bottom=0.095,
        top=0.82,
        wspace=0.14,
        hspace=0.32,
    )

    fig.suptitle("RAGBench Local Test Results", x=0.06, y=0.965, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.06,
        0.918,
        "1,000 samples; detector evaluated with gold contexts and locally retrieved contexts.",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#475569",
    )

    ax_metrics = fig.add_subplot(grid[0, :2])
    ax_confusion = fig.add_subplot(grid[:, 2])
    ax_retrieval = fig.add_subplot(grid[1, 0])
    ax_quality = fig.add_subplot(grid[1, 1])

    _plot_classification_metrics(ax_metrics, summary)
    _plot_confusion_matrices(ax_confusion, summary)
    _plot_retrieval_metrics(ax_retrieval, summary)
    _plot_quality_scores(ax_quality, summary)
    return fig


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#334155",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


def _plot_classification_metrics(ax, summary: dict) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    modes = ["oracle_context", "retrieved_context"]
    x_positions = list(range(len(metrics)))
    width = 0.36
    colors = ["#2563EB", "#F97316"]

    for index, mode in enumerate(modes):
        values = [summary[mode][metric] for metric in metrics]
        offsets = [x + (index - 0.5) * width for x in x_positions]
        bars = ax.bar(offsets, values, width=width, label=MODE_LABELS[mode], color=colors[index])
        _annotate_bars(ax, bars, values)

    ax.set_title("A. Hallucination classification", loc="left", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions, ["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.55, linewidth=0.7)
    ax.legend(loc="upper right", frameon=False)


def _plot_confusion_matrices(ax, summary: dict) -> None:
    ax.set_title("B. Confusion matrix counts", loc="left", fontweight="bold")
    ax.axis("off")

    matrices = [
        ("Oracle context", summary["oracle_context"]),
        ("Retrieved context", summary["retrieved_context"]),
    ]
    cmap = LinearSegmentedColormap.from_list("confusion_blue", ["#EFF6FF", "#2563EB"])

    for row_index, (title, values) in enumerate(matrices):
        inset = ax.inset_axes([0.08, 0.55 - row_index * 0.47, 0.84, 0.36])
        matrix = [[values["tn"], values["fp"]], [values["fn"], values["tp"]]]
        max_value = max(max(line) for line in matrix)
        inset.imshow(matrix, cmap=cmap, vmin=0, vmax=max_value)
        inset.set_title(title, fontsize=9, pad=7, color="#1E293B")
        inset.set_xticks([0, 1], ["Pred neg", "Pred pos"])
        inset.set_yticks([0, 1], ["Actual neg", "Actual pos"])
        inset.tick_params(length=0, labelsize=7.5)
        for y in range(2):
            for x in range(2):
                value = matrix[y][x]
                color = "white" if value > max_value * 0.55 else "#0F172A"
                inset.text(x, y, str(value), ha="center", va="center", color=color, fontsize=10, fontweight="bold")
        for spine in inset.spines.values():
            spine.set_visible(False)


def _plot_retrieval_metrics(ax, summary: dict) -> None:
    values = [
        summary["retrieval"]["source_hit_rate"],
        summary["averages"]["retrieved_context_precision"],
        summary["retrieval"]["avg_max_retrieved_score"],
    ]
    labels = ["Source hit", "Context precision", "Max score"]
    bars = ax.barh(labels, values, color=["#14B8A6", "#64748B", "#8B5CF6"])
    ax.set_title("C. Retrieval quality", loc="left", fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", color="#CBD5E1", alpha=0.55, linewidth=0.7)
    for bar, value in zip(bars, values, strict=False):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)


def _plot_quality_scores(ax, summary: dict) -> None:
    values = [
        summary["averages"]["oracle_hallucination_rate"],
        summary["averages"]["retrieved_hallucination_rate"],
        summary["averages"]["oracle_faithfulness"],
        summary["averages"]["retrieved_faithfulness"],
    ]
    labels = ["Oracle\nhalluc.", "Retrieved\nhalluc.", "Oracle\nfaith.", "Retrieved\nfaith."]
    colors = ["#93C5FD", "#FB923C", "#1D4ED8", "#EA580C"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("D. Average detector scores", loc="left", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.55, linewidth=0.7)
    ax.tick_params(axis="x", labelsize=8)
    _annotate_bars(ax, bars, values)


def _annotate_bars(ax, bars, values: list[float]) -> None:
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


if __name__ == "__main__":
    main()
