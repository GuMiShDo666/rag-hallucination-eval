<h1 align="center">RAG Hallucination Eval</h1>

<p align="center">
  <strong>面向 RAG 系统的本地优先幻觉检测与批量评测工具</strong>
  <br />
  <em>Claim-level detection · External RAG evaluation · Qwen-compatible API · Reproducible experiments</em>
</p>

<p align="center">
  <a href="#快速开始"><img src="https://img.shields.io/badge/Quick_Start-Run_Local-2563EB?style=for-the-badge" alt="Quick Start" /></a>
  <a href="#最小-api-示例"><img src="https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge" alt="FastAPI API" /></a>
  <a href="#运行结果展示"><img src="https://img.shields.io/badge/Results-Evaluation-7C3AED?style=for-the-badge" alt="Results" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/RAG-Evaluation-0F766E?style=flat-square" alt="RAG Evaluation" />
  <img src="https://img.shields.io/badge/Vector_Search-FAISS-2563EB?style=flat-square" alt="FAISS" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/tests-pytest-0A7F3F?style=flat-square" alt="pytest" />
</p>

<p align="center">
  <a href="README_EN.md">English</a> ·
  <a href="docs/api.md">API 文档</a> ·
  <a href="docs/datasets.md">数据集文档</a> ·
  <a href="docs/external_ragflow_eval.md">外部 RAG 测试</a>
</p>

---

RAG Hallucination Eval 用来检查 RAG 问答结果中哪些陈述被上下文支持、哪些可能是幻觉。它既可以运行一个内置 RAG demo，也可以作为独立 API 接收其他 RAG 系统的 `question + answer + contexts`，输出 claim 级检测结果和评测指标。

## 核心能力

| 能力 | 说明 |
|---|---|
| 内置 RAG 流程 | 加载 `.txt`、`.md`、`.pdf`，切分 chunk，构建 FAISS 索引并生成答案 |
| 外部 RAG 评测 | 直接检测其他 RAG 系统输出，不要求对方使用本项目的检索链路 |
| Claim 级检测 | 将答案拆成可判断片段，标注 `supported`、`unsupported`、`contradicted`、`unclear` |
| 批量评测 API | `/batch_evaluate` 支持一次提交多条样本，适合离线评测集或回归测试 |
| 可复现实验 | 提供 baseline、chunk size、top-k、query rewrite、reranker 开关实验 |
| 离线可跑 | 缺少 API key 或 embedding 模型不可用时，可通过 mock LLM 和 hashing embedding 跑通流程 |

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m pytest
```

启动 Web Demo：

```bash
streamlit run app/streamlit_app.py
```

启动 API 服务：

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

打开接口文档：

```text
http://127.0.0.1:8000/docs
```

## 最小 API 示例

检测一条外部 RAG 输出：

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

常用端点：

| Endpoint | 方法 | 用途 |
|---|---|---|
| `/health` | GET | 服务健康检查 |
| `/detect` | POST | 只做 claim 级幻觉检测 |
| `/evaluate` | POST | 单条样本检测并计算指标 |
| `/batch_evaluate` | POST | 批量检测外部 RAG 输出 |

详见 [docs/api.md](docs/api.md)。

## 架构

![RAG Hallucination Eval Architecture](docs/assets/architecture.svg)

内置 demo 和外部系统输出最终都会进入同一个检测核心。稳定输入契约是：

```text
question + answer + contexts
```

核心处理链路：

1. `src/document_loader.py` 将文档标准化为内部 `Document`。
2. `src/chunker.py` 切分文本并保留 `source`、`page`、`chunk_id`。
3. `src/embedder.py` 优先使用 sentence-transformers，失败时回退 hashing embedding。
4. `src/retriever.py` 使用 FAISS 执行 top-k 检索。
5. `src/generator.py` 通过 grounding prompt 生成带上下文约束的答案。
6. `src/hallucination_detector.py` 做 claim 级支持性判断。
7. `src/evaluator.py` 计算 faithfulness、answer relevancy、context precision、citation accuracy 和 hallucination rate。

## 配置

编辑 `.env`：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `qwen` | 可选 `qwen`、`openai`、`deepseek`、`mock` |
| `QWEN_API_KEY` | `your_key` | Qwen API key |
| `QWEN_MODEL` | `qwen-plus` | Qwen 模型名 |
| `OPENAI_API_KEY` | `your_key` | OpenAI API key |
| `DEEPSEEK_API_KEY` | `your_key` | DeepSeek API key |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | sentence-transformers 模型 |
| `VECTOR_STORE_PATH` | `data/processed/faiss_index` | FAISS 索引输出路径 |
| `DEFAULT_CHUNK_SIZE` | `512` | 默认 chunk 大小 |
| `DEFAULT_CHUNK_OVERLAP` | `80` | 默认 chunk overlap |
| `DEFAULT_TOP_K` | `5` | 默认检索数量 |
| `MOCK_LLM` | `false` | 设置为 `true` 后不调用真实 LLM API |

Qwen 示例：

```bash
LLM_PROVIDER=qwen
QWEN_API_KEY=your_qwen_api_key
QWEN_MODEL=qwen-plus
MOCK_LLM=false
```

离线测试：

```bash
MOCK_LLM=true
```

## 实验与数据集

运行 baseline：

```bash
python experiments/run_baseline.py
```

运行消融实验：

```bash
python experiments/run_ablation.py
```

生成结果图：

```bash
python experiments/plot_results.py
```

运行本地 RAGBench 测试：

```bash
python experiments/run_test.py --limit 1000
```

测试报告见 [docs/test.md](docs/test.md)。

导入外部评测集：

```bash
python scripts/import_datasets.py \
  --source ragbench \
  --input /path/to/ragbench.jsonl \
  --output data/eval_sets/ragbench_eval.json \
  --limit 1000 \
  --require-context
```

支持 `ragtruth`、`ragbench`、`halueval`、`generic` 四种导入 profile，详见 [docs/datasets.md](docs/datasets.md)。

当前仓库包含 `data/eval_sets/ragbench_covidqa_1000.json`，来源为 RAGBench `covidqa/train`，共 1000 条，其中 `supported` 858 条、`unsupported` 142 条。

## 运行结果展示

1000 条 RAGBench 本地测试输出：

```text
Evaluated 1000 rows
oracle_context:   accuracy=0.7760 precision=0.1694 recall=0.1479 f1=0.1579
retrieved_context: accuracy=0.4010 precision=0.1522 recall=0.7042 f1=0.2503
retrieval_source_hit_rate: 0.3560
Wrote results/test/per_sample.csv
Wrote results/test/summary.json
Wrote docs/test.md
```

分类指标：

| 模式 | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_context | 0.7760 | 0.1694 | 0.1479 | 0.1579 | 21 | 103 | 755 | 121 |
| retrieved_context | 0.4010 | 0.1522 | 0.7042 | 0.2503 | 100 | 557 | 301 | 42 |

检索与质量指标：

| 指标 | 结果 |
|---|---:|
| 样本数量 | 1000 |
| retrieval_source_hit_rate | 0.3560 |
| retrieved_context_precision | 0.0814 |
| avg_max_retrieved_score | 0.4463 |
| oracle_hallucination_rate | 0.0898 |
| retrieved_hallucination_rate | 0.6105 |
| oracle_faithfulness | 0.9102 |
| retrieved_faithfulness | 0.3895 |

完整报告见 [docs/test.md](docs/test.md)。

## 项目结构

```text
rag-hallucination-eval/
├── api/                         # FastAPI 服务
├── app/                         # Streamlit Web Demo
├── data/
│   ├── documents/               # 示例知识库文档
│   ├── eval_sets/               # 评测集
│   ├── imported/                # 本地导入缓存
│   └── processed/               # FAISS 索引输出
├── docs/                        # API、数据集、外部测试文档
├── experiments/                 # baseline、ablation、plot 脚本
├── scripts/                     # 数据集下载与导入脚本
├── src/                         # 核心 RAG 与幻觉检测模块
├── tests/                       # pytest 测试
├── README.md
└── README_EN.md
```

## 输出文件

| 路径 | 内容 |
|---|---|
| `results/baseline_results.csv` | baseline 评测结果 |
| `results/ablation_results.csv` | 消融实验汇总 |
| `results/ablation_runs/` | 每组消融配置的详细结果 |
| `results/figures/` | Matplotlib 输出图表 |
| `data/processed/faiss_index*` | 本地 FAISS 索引文件 |

## 验证状态

当前本地测试：

```text
32 passed, 2 warnings
```

## 已知限制

- Reranker 目前是实验开关，占位接入，尚未实现真实重排策略。
- Query rewrite 已接入 LLM provider；mock 模式或缺少 provider key 时会回退到原问题。
- fallback evaluator 基于词面重叠，不等同于强语义评测。
- Ragas、DeepEval、LettuceDetect 目前不是稳定流程的必需依赖。
- 项目当前没有检测到 `LICENSE` 文件。
