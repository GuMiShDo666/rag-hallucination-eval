# Dataset Import Guide

This project uses a compact JSON evaluation format:

```json
{
  "question": "...",
  "reference_answer": "...",
  "gold_context": "...",
  "answerable": true,
  "risk_type": "unsupported",
  "source_dataset": "ragtruth",
  "candidate_answer": "...",
  "supporting_chunk_ids": ["..."],
  "expected_citations": ["..."]
}
```

Required fields for the current evaluator:

| Field | Required | Description |
|---|---:|---|
| `question` | yes | User question sent to the RAG pipeline |
| `reference_answer` | no | Gold answer used by fallback answer relevancy |
| `gold_context` | no | Evidence text used by fallback context precision |
| `answerable` | no | Whether the provided context contains an answer |
| `risk_type` | no | Label such as `unsupported`, `groundedness`, or `hallucinated_answer` |
| `source_dataset` | no | Original dataset name |
| `candidate_answer` | no | Pre-generated answer from hallucination datasets |
| `supporting_chunk_ids` | no | Gold evidence IDs if available |
| `expected_citations` | no | Gold citation IDs if available |

## Import Commands

The importer accepts `.json`, `.jsonl`, and `.csv` files:

```bash
python scripts/import_datasets.py \
  --source ragbench \
  --input /path/to/ragbench.jsonl \
  --output data/imported/ragbench_eval.json \
  --limit 1000 \
  --require-context
```

RAGTruth-style input:

```bash
python scripts/import_datasets.py \
  --source ragtruth \
  --input /path/to/ragtruth.csv \
  --output data/imported/ragtruth_eval.json \
  --require-context
```

HaluEval-style input:

```bash
python scripts/import_datasets.py \
  --source halueval \
  --input /path/to/halueval_qa.json \
  --output data/imported/halueval_eval.json \
  --limit 1000
```

Generic input:

```bash
python scripts/import_datasets.py \
  --source generic \
  --input /path/to/custom_eval.jsonl \
  --output data/imported/custom_eval.json
```

Use `--append` to add rows to an existing imported JSON file.

## Supported Source Profiles

| Source | Input purpose | Common fields handled |
|---|---|---|
| `ragtruth` | RAG hallucination annotations | `query`, `response`, `source_info`, `hallucination_type` |
| `ragbench` | RAG groundedness and faithfulness data | `question`, `response`, `documents`, `label` |
| `halueval` | Right answer vs hallucinated answer pairs | `question`, `right_answer`, `hallucinated_answer`, `knowledge` |
| `generic` | Custom local datasets | `question`, `reference_answer`, `gold_context`, `contexts` |

Imported files are written under `data/imported/` by default.
