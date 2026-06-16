# External RAG Project Smoke Test: RAGFlow

## Project Selected

| Field | Value |
|---|---|
| Repository | `infiniflow/ragflow` |
| URL | https://github.com/infiniflow/ragflow |
| Stars checked | 82,937 |
| Reason | RAGFlow is an explicit open-source RAG engine with Agent capabilities. |

Compared repositories:

| Repository | Stars checked | Notes |
|---|---:|---|
| `langgenius/dify` | 145,472 | Larger platform for agentic workflows |
| `infiniflow/ragflow` | 82,937 | Explicit RAG engine, selected |
| `Mintplex-Labs/anything-llm` | 61,664 | Local-first agent/RAG experience |
| `QuivrHQ/quivr` | 39,166 | RAG integration project |

## Method

Full RAGFlow deployment requires Docker, service initialization, model configuration, and storage components. For this smoke test, the public RAGFlow README was used as the source context, and three external-RAG-style outputs were submitted to this project's `/batch_evaluate` API.

The test validates that this project's batch API can evaluate outputs from an external RAG system shape:

```json
{
  "question": "...",
  "answer": "...",
  "contexts": ["..."]
}
```

## Batch API Result

```json
{
  "summary": {
    "total": 3,
    "supported": 2,
    "partially_supported": 0,
    "unsupported": 1,
    "avg_hallucination_rate": 0.3333333333333333,
    "avg_faithfulness": 0.6666666666666666,
    "avg_answer_relevancy": 0.6309523809523809,
    "avg_context_precision": 1.0,
    "avg_citation_accuracy": null
  }
}
```

## Cases

| ID | Expected | Result | Notes |
|---|---|---|---|
| `ragflow-supported-overview` | supported | supported | The answer matched the README description of RAGFlow as an open-source RAG engine with Agent capabilities. |
| `ragflow-supported-features` | supported | supported | The answer matched the README data-source support list. |
| `ragflow-unsupported-fabrication` | unsupported | unsupported | The answer claimed RAGFlow was created by NASA for Mars rover navigation, which was not supported by the provided context. |

## Limitation

This is an API integration smoke test using RAGFlow public documentation as context. It is not a full deployment benchmark of a running RAGFlow instance.
