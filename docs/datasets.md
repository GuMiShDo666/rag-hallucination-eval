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
| `source_name` | no | Original subset/split name if available |

## Import Commands

The importer accepts `.json`, `.jsonl`, `.csv`, and `.parquet` files:

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
| `ragbench` | RAG groundedness and faithfulness data | `question`, `response`, `documents`, `adherence_score`, `unsupported_response_sentence_keys` |
| `halueval` | Right answer vs hallucinated answer pairs | `question`, `right_answer`, `hallucinated_answer`, `knowledge` |
| `generic` | Custom local datasets | `question`, `reference_answer`, `gold_context`, `contexts` |

Imported files are written under `data/imported/` by default.

## Build a 1000-Row Local Eval Set

RAGBench is published as `galileo-ai/ragbench` on Hugging Face under `cc-by-4.0`. The helper below downloads the `covidqa/train` parquet split and writes the first 1000 normalized rows to `data/eval_set_1000.json`:

```bash
python scripts/download_ragbench_sample.py \
  --subset covidqa \
  --split train \
  --limit 1000 \
  --output data/eval_set_1000.json
```

The checked-in `data/eval_set_1000.json` file was generated from RAGBench `covidqa/train` and contains:

| Field | Value |
|---|---:|
| rows | 1000 |
| supported | 858 |
| unsupported | 142 |
| rows with `gold_context` | 1000 |
| rows with `supporting_chunk_ids` | 979 |

Use it with the baseline runner:

```bash
MOCK_LLM=true python experiments/run_baseline.py \
  --eval data/eval_set_1000.json \
  --output results/baseline_1000_results.csv
```
