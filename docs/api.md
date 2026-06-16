# Hallucination Detection API

Run the API server:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Open the interactive docs:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/detect` | Claim-level hallucination detection |
| `POST` | `/evaluate` | Detection plus fallback metrics |
| `POST` | `/batch_evaluate` | Batch evaluation for external RAG outputs |

## Detect

Request:

```bash
curl -s http://127.0.0.1:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What problem does LoRA solve?",
    "answer": "LoRA reduces trainable parameters by injecting low-rank matrices. [1]",
    "contexts": [
      "LoRA reduces trainable parameters by injecting low-rank matrices into model weights."
    ]
  }'
```

Response:

```json
{
  "spans": [
    {
      "text": "LoRA reduces trainable parameters by injecting low-rank matrices. [1]",
      "status": "supported",
      "reason": "Most claim terms appear in the retrieved context.",
      "confidence": 0.875
    }
  ],
  "unsupported_spans": [],
  "hallucination_rate": 0.0,
  "final_judgement": "supported"
}
```

## Evaluate

Request:

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

Response adds:

```json
{
  "faithfulness": 1.0,
  "answer_relevancy": 0.6,
  "context_precision": 1.0,
  "citation_accuracy": 1.0
}
```

Set `use_llm_judge` to `true` in the request body to use Qwen as the judge when `QWEN_API_KEY` is configured.

## Batch Evaluate

Request:

```bash
curl -s http://127.0.0.1:8000/batch_evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "case-1",
        "question": "What problem does LoRA solve?",
        "answer": "LoRA reduces trainable parameters. [1]",
        "contexts": [
          "LoRA reduces trainable parameters during fine-tuning."
        ],
        "reference_answer": "LoRA reduces trainable parameters.",
        "gold_context": "LoRA reduces trainable parameters during fine-tuning."
      },
      {
        "id": "case-2",
        "question": "What problem does LoRA solve?",
        "answer": "The Amazon rainforest is home to many species.",
        "contexts": [
          "LoRA reduces trainable parameters during fine-tuning."
        ],
        "reference_answer": "LoRA reduces trainable parameters.",
        "gold_context": "LoRA reduces trainable parameters during fine-tuning."
      }
    ]
  }'
```

Response:

```json
{
  "summary": {
    "total": 2,
    "supported": 1,
    "partially_supported": 0,
    "unsupported": 1,
    "avg_hallucination_rate": 0.5,
    "avg_faithfulness": 0.5,
    "avg_answer_relevancy": 0.5,
    "avg_context_precision": 1.0,
    "avg_citation_accuracy": 1.0
  },
  "results": []
}
```
