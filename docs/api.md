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
