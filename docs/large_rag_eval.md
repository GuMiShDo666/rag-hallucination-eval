# Large Local RAG Evaluation

This experiment builds a local RAG corpus from `data/eval_sets/ragbench_covidqa_1000.json` and evaluates 1000 RAGBench candidate answers against the project hallucination detector.

## Setup

| Setting | Value |
|---|---:|
| Samples | 1000 |
| Chunk size | 1200 |
| Chunk overlap | 120 |
| Top-k | 5 |
| Supported rows | 858 |
| Unsupported rows | 142 |

Two modes were measured:

- `oracle_context`: detect `candidate_answer` against the gold context.
- `retrieved_context`: retrieve contexts from the local FAISS corpus first, then detect.

## Hallucination Detection Metrics

| Mode | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_context | 0.7760 | 0.1694 | 0.1479 | 0.1579 | 21 | 103 | 755 | 121 |
| retrieved_context | 0.4010 | 0.1522 | 0.7042 | 0.2503 | 100 | 557 | 301 | 42 |

## Retrieval Metrics

| Metric | Value |
|---|---:|
| source_hit_rate | 0.3560 |
| avg_max_retrieved_score | 0.4463 |
| retrieved_context_precision | 0.0814 |

## Average Scores

| Metric | Value |
|---|---:|
| oracle_hallucination_rate | 0.0898 |
| retrieved_hallucination_rate | 0.6105 |
| oracle_faithfulness | 0.9102 |
| retrieved_faithfulness | 0.3895 |

## Interpretation

The previous 100% results came from the tiny 5-row sample set and mock generation that copied heavily from retrieved context. On the 1000-row RAGBench sample, the current local fallback detector has high overall accuracy mainly because supported answers dominate the dataset, but it has low precision and recall for hallucinated or unsupported answers.

The main bottleneck is the detector: it relies on lexical overlap, so it misses many unsupported answers that reuse terms from the context, and it can flag supported answers when wording differs from the gold context.

## Output Files

| File | Description |
|---|---|
| `results/large_rag_eval/per_sample.csv` | Per-sample predictions and metrics |
| `results/large_rag_eval/summary.json` | Aggregate metrics JSON |
