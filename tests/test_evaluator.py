import json

from src.evaluator import RAGEvaluator
from src.retriever import RetrievedChunk


def test_evaluate_single_returns_fallback_metrics():
    evaluator = RAGEvaluator()
    contexts = ["LoRA reduces trainable parameters by injecting low-rank matrices into model weights."]
    answer = "LoRA reduces trainable parameters by injecting low-rank matrices. [1]"

    result = evaluator.evaluate_single(
        question="What problem does LoRA solve?",
        answer=answer,
        contexts=contexts,
        reference_answer="LoRA reduces the number of trainable parameters for fine-tuning large language models.",
        gold_context="LoRA reduces trainable parameters by injecting low-rank matrices into model weights.",
    )

    assert result.faithfulness == 1.0
    assert result.hallucination_rate == 0.0
    assert result.context_precision == 1.0
    assert result.citation_accuracy == 1.0
    assert result.raw["unsupported_spans"] == []


def test_evaluate_single_handles_missing_gold_context_and_bad_citation():
    evaluator = RAGEvaluator()
    contexts = ["RAG retrieves context."]

    result = evaluator.evaluate_single(
        question="Why is RAG used?",
        answer="RAG retrieves context. [2]",
        contexts=contexts,
    )

    assert result.context_precision is None
    assert result.citation_accuracy == 0.0


def test_evaluate_batch_saves_expected_csv(tmp_path):
    eval_path = tmp_path / "eval_set.json"
    output_path = tmp_path / "results.csv"
    eval_path.write_text(
        json.dumps(
            [
                {
                    "question": "Why is RAG used?",
                    "reference_answer": "RAG grounds answers in external documents.",
                    "gold_context": "RAG combines retrieval with generation to ground answers in external documents.",
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakePipeline:
        def ask(self, question):
            return {
                "question": question,
                "answer": "RAG combines retrieval with generation to ground answers in external documents. [1]",
                "contexts": [
                    RetrievedChunk(
                        text="RAG combines retrieval with generation to ground answers in external documents.",
                        score=1.0,
                        source="notes.md",
                        page=None,
                        chunk_id="notes_pnone_c0",
                        metadata={},
                    )
                ],
            }

    dataframe = RAGEvaluator().evaluate_batch(str(eval_path), FakePipeline(), str(output_path))

    assert output_path.exists()
    assert list(dataframe.columns) == [
        "question",
        "answer",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "hallucination_rate",
        "citation_accuracy",
        "retrieved_chunk_ids",
        "unsupported_spans",
    ]
    assert dataframe.loc[0, "retrieved_chunk_ids"] == "notes_pnone_c0"
