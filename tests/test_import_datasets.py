from scripts.import_datasets import import_dataset


def test_import_ragbench_jsonl_fixture():
    rows = import_dataset(
        source="ragbench",
        input_path="tests/fixtures/datasets/ragbench_sample.jsonl",
    )

    assert rows == [
        {
            "question": "Why is RAG used?",
            "reference_answer": "RAG grounds answers in retrieved documents.",
            "gold_context": "RAG combines retrieval with generation to ground answers in external documents.",
            "answerable": True,
            "risk_type": "grounded",
            "source_dataset": "ragbench",
            "candidate_answer": "RAG grounds answers in retrieved documents.",
            "source_id": "rb-1",
        }
    ]


def test_import_halueval_json_fixture_keeps_hallucinated_answer():
    rows = import_dataset(
        source="halueval",
        input_path="tests/fixtures/datasets/halueval_sample.json",
    )

    assert rows[0]["question"] == "What problem does LoRA solve?"
    assert rows[0]["reference_answer"] == "LoRA reduces trainable parameters during fine-tuning."
    assert rows[0]["candidate_answer"] == "LoRA increases the number of trainable parameters."
    assert rows[0]["risk_type"] == "hallucinated_answer"
    assert rows[0]["answerable"] is True


def test_import_ragtruth_csv_fixture():
    rows = import_dataset(
        source="ragtruth",
        input_path="tests/fixtures/datasets/ragtruth_sample.csv",
    )

    assert rows[0]["question"] == "Does RAG guarantee factual correctness?"
    assert rows[0]["reference_answer"] == "RAG does not automatically guarantee factual correctness."
    assert rows[0]["gold_context"] == "RAG can still produce unsupported claims if retrieval is incomplete."
    assert rows[0]["risk_type"] == "unsupported"


def test_import_limit_and_require_context():
    rows = import_dataset(
        source="ragbench",
        input_path="tests/fixtures/datasets/ragbench_sample.jsonl",
        limit=1,
        require_context=True,
    )

    assert len(rows) == 1
