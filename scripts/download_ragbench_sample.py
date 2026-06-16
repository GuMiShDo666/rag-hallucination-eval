"""Download a RAGBench split sample and convert it to the project eval format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

from import_datasets import import_dataset


RAGBENCH_BASE_URL = "https://huggingface.co/datasets/galileo-ai/ragbench/resolve/main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize a RAGBench sample.")
    parser.add_argument("--subset", default="covidqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", default="data/eval_sets/ragbench_covidqa_1000.json")
    parser.add_argument("--cache-dir", default="data/imported/cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = _download_parquet(args.subset, args.split, args.cache_dir)
    rows = import_dataset(
        source="ragbench",
        input_path=str(parquet_path),
        limit=args.limit,
        require_context=True,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} rows to {output_path}")
    print(f"Source: galileo-ai/ragbench {args.subset}/{args.split}")


def _download_parquet(subset: str, split: str, cache_dir: str) -> Path:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    filename = f"{split}-00000-of-00001.parquet"
    output_path = cache_path / subset / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    url = f"{RAGBENCH_BASE_URL}/{subset}/{filename}"
    print(f"Downloading {url}")
    urlretrieve(url, output_path)
    return output_path


if __name__ == "__main__":
    main()
