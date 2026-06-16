# Data Layout

```text
data/
├── documents/          # Source documents indexed by the RAG pipeline
├── eval_sets/          # Versioned evaluation sets used by experiments
├── imported/           # Local download cache and temporary imports
└── processed/          # Generated FAISS indexes and processed artifacts
```

## Evaluation Sets

| File | Rows | Source | Purpose |
|---|---:|---|---|
| `eval_sets/sample.json` | 5 | Hand-written local sample | Fast smoke tests and examples |
| `eval_sets/ragbench_covidqa_1000.json` | 1000 | RAGBench `covidqa/train` | Larger groundedness and hallucination evaluation |

Use explicit names for new eval sets:

```text
eval_sets/<source>_<subset>_<rows>.json
```

Examples:

```text
eval_sets/ragbench_covidqa_1000.json
eval_sets/halueval_qa_1000.json
eval_sets/ragtruth_main_1000.json
```

Downloaded parquet files and intermediate imports should stay under `data/imported/`.
