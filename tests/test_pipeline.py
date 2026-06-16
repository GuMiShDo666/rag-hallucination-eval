import numpy as np

from src.pipeline import RAGHallucinationPipeline


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "lora" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "rag" in lowered or "retrieval" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeQueryRewriter:
    def rewrite(self, question):
        class Result:
            rewritten_query = "RAG retrieval grounded generation"

        return Result()


def test_pipeline_initializes(tmp_path):
    pipeline = RAGHallucinationPipeline(
        docs_path=str(tmp_path),
        embedder=FakeEmbedder(),
    )

    assert pipeline.docs_path == str(tmp_path)
    assert pipeline.top_k == 5
    assert pipeline.is_built is False


def test_pipeline_returns_answer_with_mock_llm(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.md").write_text(
        "RAG combines retrieval with generation to ground answers in external documents.\n\n"
        "LoRA reduces trainable parameters by injecting low-rank matrices into model weights.",
        encoding="utf-8",
    )
    pipeline = RAGHallucinationPipeline(
        docs_path=str(docs_dir),
        chunk_size=120,
        chunk_overlap=10,
        top_k=1,
        embedder=FakeEmbedder(),
    )
    pipeline.generator.mock_mode = True

    pipeline.build()
    result = pipeline.ask("Why is RAG used in question answering?")

    assert result["question"] == "Why is RAG used in question answering?"
    assert result["answer"]
    assert result["contexts"]
    assert result["retrieval_query"] == result["question"]
    assert result["query_rewritten"] is False
    assert "hallucination_rate" in result
    assert "final_judgement" in result


def test_pipeline_uses_query_rewriter_for_retrieval(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.md").write_text(
        "RAG combines retrieval with generation to ground answers in external documents.\n\n"
        "LoRA reduces trainable parameters by injecting low-rank matrices into model weights.",
        encoding="utf-8",
    )
    pipeline = RAGHallucinationPipeline(
        docs_path=str(docs_dir),
        chunk_size=120,
        chunk_overlap=10,
        top_k=1,
        use_query_rewrite=True,
        embedder=FakeEmbedder(),
        query_rewriter=FakeQueryRewriter(),
    )
    pipeline.generator.mock_mode = True

    pipeline.build()
    result = pipeline.ask("How does grounding work?")

    assert result["retrieval_query"] == "RAG retrieval grounded generation"
    assert result["query_rewritten"] is True
    assert "RAG combines retrieval" in result["contexts"][0]["text"]


def test_pipeline_requires_build_before_ask(tmp_path):
    pipeline = RAGHallucinationPipeline(
        docs_path=str(tmp_path),
        embedder=FakeEmbedder(),
    )

    try:
        pipeline.ask("Why is RAG used?")
    except ValueError as exc:
        assert "build" in str(exc)
    else:
        raise AssertionError("ask should fail before build")
