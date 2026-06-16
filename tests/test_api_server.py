from fastapi.testclient import TestClient

from api.server import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_detect_endpoint_returns_unsupported_spans():
    response = client.post(
        "/detect",
        json={
            "question": "What problem does LoRA solve?",
            "answer": "The Eiffel Tower is located in Berlin.",
            "contexts": [
                "LoRA reduces trainable parameters by injecting low-rank matrices into model weights."
            ],
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["hallucination_rate"] == 1.0
    assert payload["unsupported_spans"]
    assert payload["final_judgement"] == "unsupported"


def test_evaluate_endpoint_returns_metrics():
    response = client.post(
        "/evaluate",
        json={
            "question": "What problem does LoRA solve?",
            "answer": "LoRA reduces trainable parameters by injecting low-rank matrices. [1]",
            "contexts": [
                "LoRA reduces trainable parameters by injecting low-rank matrices into model weights."
            ],
            "reference_answer": "LoRA reduces trainable parameters during fine-tuning.",
            "gold_context": "LoRA reduces trainable parameters by injecting low-rank matrices into model weights.",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["faithfulness"] == 1.0
    assert payload["context_precision"] == 1.0
    assert payload["citation_accuracy"] == 1.0
