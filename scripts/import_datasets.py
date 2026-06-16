"""Normalize external RAG hallucination datasets into this project's eval format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


SOURCE_ALIASES = {
    "question": [
        "question",
        "query",
        "prompt",
        "user_query",
        "instruction",
    ],
    "reference_answer": [
        "reference_answer",
        "ground_truth",
        "ground_truth_answer",
        "gold_answer",
        "right_answer",
        "answer",
        "response",
        "output",
    ],
    "candidate_answer": [
        "candidate_answer",
        "hallucinated_answer",
        "model_answer",
        "model_response",
        "generated_answer",
        "response",
        "output",
    ],
    "gold_context": [
        "gold_context",
        "context",
        "contexts",
        "documents",
        "passages",
        "evidence",
        "knowledge",
        "source_info",
        "source",
    ],
    "supporting_chunk_ids": [
        "supporting_chunk_ids",
        "supporting_context_ids",
        "evidence_ids",
        "relevant_doc_ids",
        "all_relevant_sentence_keys",
    ],
    "risk_type": [
        "risk_type",
        "hallucination_type",
        "label",
        "category",
        "error_type",
    ],
}


SOURCE_DEFAULTS = {
    "ragtruth": {
        "risk_type": "rag_hallucination",
    },
    "ragbench": {
        "risk_type": "groundedness",
    },
    "halueval": {
        "risk_type": "hallucinated_answer",
    },
    "generic": {
        "risk_type": "unknown",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import external RAG evaluation datasets.")
    parser.add_argument("--source", choices=sorted(SOURCE_DEFAULTS), required=True)
    parser.add_argument("--input", required=True, help="Input .json, .jsonl, or .csv file.")
    parser.add_argument("--output", default="data/imported/eval_set_imported.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true", help="Append imported rows to an existing output file.")
    parser.add_argument("--require-context", action="store_true", help="Drop rows without a usable context field.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imported = import_dataset(
        source=args.source,
        input_path=args.input,
        limit=args.limit,
        require_context=args.require_context,
    )
    output_path = Path(args.output)
    if args.append and output_path.exists():
        existing = _load_json(output_path)
        if not isinstance(existing, list):
            raise ValueError(f"Existing output must be a JSON list: {output_path}")
        imported = [*existing, *imported]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(imported, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(imported)} rows to {output_path}")


def import_dataset(
    source: str,
    input_path: str,
    limit: int | None = None,
    require_context: bool = False,
) -> list[dict]:
    """Load and normalize rows from a supported external dataset source."""

    if source not in SOURCE_DEFAULTS:
        raise ValueError(f"Unsupported source: {source}")

    rows = load_rows(input_path)
    normalized: list[dict] = []
    for row in rows:
        item = normalize_row(row, source=source)
        if not item:
            continue
        if require_context and not item.get("gold_context"):
            continue
        normalized.append(item)
        if limit is not None and len(normalized) >= limit:
            break
    return normalized


def load_rows(input_path: str) -> list[dict]:
    """Load rows from JSON, JSONL, CSV, or Parquet input."""

    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _load_json(path)
        if isinstance(payload, list):
            return [_coerce_dict(row) for row in payload]
        if isinstance(payload, dict):
            for key in ["data", "rows", "examples", "items"]:
                value = payload.get(key)
                if isinstance(value, list):
                    return [_coerce_dict(row) for row in value]
            return [payload]
        raise ValueError(f"JSON input must be an object or list: {path}")

    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(_coerce_dict(json.loads(line)))
        return rows

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]

    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Parquet import requires pandas and pyarrow.") from exc
        dataframe = pd.read_parquet(path)
        return dataframe.to_dict(orient="records")

    raise ValueError(f"Unsupported input format: {path.suffix}")


def normalize_row(row: dict, source: str) -> dict | None:
    """Convert one external dataset row into the project eval-set schema."""

    question = _pick_text(row, SOURCE_ALIASES["question"])
    if not question:
        return None

    reference_answer = _reference_answer(row, source)
    candidate_answer = _candidate_answer(row, source)
    gold_context = _context_text(row)
    risk_type = _source_risk_type(row, source)
    supporting_chunk_ids = _pick_list(row, SOURCE_ALIASES["supporting_chunk_ids"])
    expected_citations = _pick_list(row, ["expected_citations", "citations", "citation_ids"])

    item = {
        "question": question,
        "reference_answer": reference_answer or candidate_answer or "",
        "gold_context": gold_context,
        "answerable": bool(gold_context),
        "risk_type": risk_type,
        "source_dataset": source,
    }
    if candidate_answer:
        item["candidate_answer"] = candidate_answer
    if supporting_chunk_ids:
        item["supporting_chunk_ids"] = supporting_chunk_ids
    if expected_citations:
        item["expected_citations"] = expected_citations
    if "id" in row:
        item["source_id"] = str(row["id"])
    if "dataset_name" in row:
        item["source_name"] = str(row["dataset_name"])
    return item


def _source_risk_type(row: dict, source: str) -> str:
    explicit = _pick_text(row, SOURCE_ALIASES["risk_type"])
    if explicit:
        return explicit
    if source == "ragbench":
        unsupported = _pick_list(row, ["unsupported_response_sentence_keys"])
        if unsupported:
            return "unsupported"
        adherence = _pick_value(row, ["adherence_score"])
        if isinstance(adherence, bool):
            return "supported" if adherence else "unsupported"
    return SOURCE_DEFAULTS[source]["risk_type"]


def _reference_answer(row: dict, source: str) -> str:
    if source == "halueval":
        return _pick_text(row, ["right_answer", "reference_answer", "ground_truth", "answer"])
    return _pick_text(row, SOURCE_ALIASES["reference_answer"])


def _candidate_answer(row: dict, source: str) -> str:
    if source == "halueval":
        return _pick_text(row, ["hallucinated_answer", "candidate_answer", "model_answer"])
    return _pick_text(row, SOURCE_ALIASES["candidate_answer"])


def _context_text(row: dict) -> str:
    value = _pick_value(row, SOURCE_ALIASES["gold_context"])
    if value is None:
        return ""
    return _stringify_context(value)


def _pick_text(row: dict, keys: Iterable[str]) -> str:
    value = _pick_value(row, keys)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return _stringify_context(value)


def _pick_list(row: dict, keys: Iterable[str]) -> list[str]:
    value = _pick_value(row, keys)
    if value is None:
        return []
    if _is_array_like(value):
        value = value.tolist()
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [part.strip() for part in stripped.split("|") if part.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        return [str(parsed)]
    return [str(value)]


def _pick_value(row: dict, keys: Iterable[str]) -> Any:
    lowered = {str(key).lower(): key for key in row}
    for key in keys:
        original_key = lowered.get(key.lower())
        if original_key is not None:
            value = row[original_key]
            if not _is_empty_value(value):
                return value
    return None


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple)) or _is_array_like(value):
        return len(value) == 0
    try:
        import pandas as pd

        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return True
    except (ImportError, TypeError, ValueError):
        pass
    return False


def _stringify_context(value: Any) -> str:
    if _is_array_like(value):
        value = value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            return _stringify_context(parsed)
        return stripped
    if isinstance(value, list):
        parts = [_stringify_context(item) for item in value]
        return "\n\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ["text", "content", "document", "passage", "context"]:
            if key in value:
                return _stringify_context(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict))


def _coerce_dict(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Each dataset row must be a JSON object or CSV row.")
    return value


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
