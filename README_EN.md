<h1 align="center">RAG Hallucination Eval</h1>

<p align="center">
  <strong>Local-first hallucination detection and batch evaluation for RAG systems</strong>
  <br />
  <em>Claim-level detection · External RAG evaluation · Qwen-compatible API · Reproducible experiments</em>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-Run_Local-2563EB?style=for-the-badge" alt="Quick Start" /></a>
  <a href="#minimal-api-example"><img src="https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge" alt="FastAPI API" /></a>
  <a href="#results"><img src="https://img.shields.io/badge/Results-Evaluation-7C3AED?style=for-the-badge" alt="Results" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/RAG-Evaluation-0F766E?style=flat-square" alt="RAG Evaluation" />
  <img src="https://img.shields.io/badge/Vector_Search-FAISS-2563EB?style=flat-square" alt="FAISS" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/tests-pytest-0A7F3F?style=flat-square" alt="pytest" />
</p>

<p align="center">
  <a href="README.md">中文</a> ·
  <a href="docs/api.md">API Docs</a> ·
  <a href="docs/datasets.md">Dataset Docs</a> ·
  <a href="docs/external_ragflow_eval.md">External RAG Test</a>
</p>

---

RAG Hallucination Eval checks whether generated RAG answers are supported by their retrieved contexts. It can run as a built-in demo pipeline, or as a standalone API that evaluates outputs from any external RAG system through the stable `question + answer + contexts` contract.

## Key Features

| Feature | Description |
|---|---|
| Built-in RAG pipeline | Loads `.txt`, `.md`, and `.pdf`, chunks text, builds a FAISS index, and generates answers |
| External RAG evaluation | Evaluates outputs from other RAG systems without requiring them to use this retrieval stack |
| Claim-level detection | Labels answer spans as `supported`, `unsupported`, `contradicted`, or `unclear` |
| Batch API | `/batch_evaluate` accepts multiple samples for offline evaluation or regression tests |
| Reproducible experiments | Includes baseline, chunk size, top-k, query rewrite, and reranker-switch experiments |
| Offline runnable | Falls back to mock LLM output and hashing embeddings when API keys or embedding downloads are unavailable |

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m pytest
```

Start the Streamlit demo:

```bash
streamlit run app/streamlit_app.py
```

Start the API server:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## Minimal API Example

Evaluate one external RAG answer:

```bash
curl -s http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What problem does LoRA solve?",
    "answer": "LoRA reduces trainable parameters by injecting low-rank matrices. [1]",
    "contexts": [
      "LoRA reduces trainable parameters by injecting low-rank matrices into model weights."
    ],
    "reference_answer": "LoRA reduces trainable parameters during fine-tuning.",
    "gold_context": "LoRA reduces trainable parameters by injecting low-rank matrices into model weights."
}'
```

Common endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service health check |
| `/detect` | POST | Claim-level hallucination detection only |
| `/evaluate` | POST | Single-sample detection and metrics |
| `/batch_evaluate` | POST | Batch evaluation for external RAG outputs |

See [docs/api.md](docs/api.md).

## Architecture

![RAG Hallucination Eval Architecture](docs/assets/architecture.svg)

The built-in demo and external RAG outputs both flow into the same detection core. The stable input contract is:

```text
question + answer + contexts
```

Core pipeline:

1. `src/document_loader.py` normalizes source files into `Document` objects.
2. `src/chunker.py` creates chunks and carries `source`, `page`, and `chunk_id` metadata.
3. `src/embedder.py` tries sentence-transformers first and falls back to hashing embeddings.
4. `src/retriever.py` performs top-k retrieval with FAISS.
5. `src/generator.py` produces context-grounded answers through a fixed prompt.
6. `src/hallucination_detector.py` performs claim-level support detection.
7. `src/evaluator.py` computes faithfulness, answer relevancy, context precision, citation accuracy, and hallucination rate.

## Configuration

Edit `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `qwen` | One of `qwen`, `openai`, `deepseek`, `mock` |
| `QWEN_API_KEY` | `your_key` | Qwen API key |
| `QWEN_MODEL` | `qwen-plus` | Qwen model name |
| `OPENAI_API_KEY` | `your_key` | OpenAI API key |
| `DEEPSEEK_API_KEY` | `your_key` | DeepSeek API key |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers model |
| `VECTOR_STORE_PATH` | `data/processed/faiss_index` | FAISS index output path |
| `DEFAULT_CHUNK_SIZE` | `512` | Default chunk size |
| `DEFAULT_CHUNK_OVERLAP` | `80` | Default chunk overlap |
| `DEFAULT_TOP_K` | `5` | Default retrieval depth |
| `MOCK_LLM` | `false` | Set to `true` to avoid real LLM API calls |

Qwen example:

```bash
LLM_PROVIDER=qwen
QWEN_API_KEY=your_qwen_api_key
QWEN_MODEL=qwen-plus
MOCK_LLM=false
```

Local-only mode:

```bash
MOCK_LLM=true
```

## Experiments and Datasets

Run the baseline:

```bash
python experiments/run_baseline.py
```

Run ablations:

```bash
python experiments/run_ablation.py
```

Generate plots:

```bash
python experiments/plot_results.py
```

Run the 1000-row local RAGBench stress test:

```bash
python experiments/run_large_rag_eval.py --limit 1000
```

See [docs/large_rag_eval.md](docs/large_rag_eval.md) for the stress-test report.

Import an external evaluation dataset:

```bash
python scripts/import_datasets.py \
  --source ragbench \
  --input /path/to/ragbench.jsonl \
  --output data/eval_sets/ragbench_eval.json \
  --limit 1000 \
  --require-context
```

Supported import profiles are `ragtruth`, `ragbench`, `halueval`, and `generic`. See [docs/datasets.md](docs/datasets.md).

This repository includes `data/eval_sets/ragbench_covidqa_1000.json`, sourced from RAGBench `covidqa/train`. It contains 1000 rows: 858 `supported` rows and 142 `unsupported` rows.

## Results

Baseline output:

```text
Saved 5 rows to results/baseline_results.csv
avg_faithfulness: 1.0000
avg_answer_relevancy: 0.3919
avg_context_precision: 0.4000
avg_hallucination_rate: 0.0000
```

Baseline metrics:

| Metric | Value |
|---|---:|
| Samples | 5 |
| avg_faithfulness | 1.0000 |
| avg_answer_relevancy | 0.3919 |
| avg_context_precision | 0.4000 |
| avg_hallucination_rate | 0.0000 |

Ablation summary:

| Setting | Faithfulness | Answer Relevancy | Context Precision | Hallucination Rate |
|---|---:|---:|---:|---:|
| chunk_size_256 | 1.0000 | 0.2417 | 0.2800 | 0.0000 |
| chunk_size_512 | 1.0000 | 0.3919 | 0.4000 | 0.0000 |
| chunk_size_1024 | 1.0000 | 0.3253 | 1.0000 | 0.0000 |
| top_k_3 | 1.0000 | 0.3919 | 0.4000 | 0.0000 |
| top_k_5 | 1.0000 | 0.3919 | 0.4000 | 0.0000 |
| top_k_8 | 1.0000 | 0.3919 | 0.4000 | 0.0000 |

Generated plots:

![Chunk Size Hallucination Rate](docs/assets/chunk_size_hallucination_rate.png)

![Top-K Faithfulness](docs/assets/top_k_faithfulness.png)

![Baseline vs Reranker](docs/assets/baseline_vs_reranker.png)

![Baseline vs Query Rewrite](docs/assets/baseline_vs_query_rewrite.png)

## Directory Layout

```text
rag-hallucination-eval/
├── api/                         # FastAPI service
├── app/                         # Streamlit Web Demo
├── data/
│   ├── documents/               # Sample knowledge-base documents
│   ├── eval_sets/               # Evaluation datasets
│   ├── imported/                # Local import cache
│   └── processed/               # FAISS index outputs
├── docs/                        # API, dataset, and external test docs
├── experiments/                 # Baseline, ablation, and plotting scripts
├── scripts/                     # Dataset download and import scripts
├── src/                         # Core RAG and hallucination detection modules
├── tests/                       # pytest tests
├── README.md
└── README_EN.md
```

## Outputs

| Path | Description |
|---|---|
| `results/baseline_results.csv` | Baseline evaluation output |
| `results/ablation_results.csv` | Ablation summary |
| `results/ablation_runs/` | Per-run ablation details |
| `results/figures/` | Generated Matplotlib figures |
| `data/processed/faiss_index*` | Local FAISS index files |

## Validation

Latest local test run:

```text
32 passed, 2 warnings
```

## Limitations

- The reranker switch is currently a placeholder for a future reranking strategy.
- Query rewriting uses the configured LLM provider and falls back to the original question in mock mode or when provider keys are missing.
- The fallback evaluator uses lexical overlap and is not a high-precision semantic judge.
- Ragas, DeepEval, and LettuceDetect are not required by the stable local workflow.
- No `LICENSE` file is currently included.
