import numpy as np

from src.chunker import Chunk
from src.retriever import FaissRetriever


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


def test_retriever_can_build_index_and_retrieve():
    chunks = [
        Chunk("LoRA reduces trainable parameters.", "notes.md", None, "notes_pnone_c0"),
        Chunk("RAG grounds answers in retrieved documents.", "notes.md", None, "notes_pnone_c1"),
    ]
    retriever = FaissRetriever(embedder=FakeEmbedder())

    retriever.build_index(chunks)
    results = retriever.retrieve("What does RAG use?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "notes_pnone_c1"
    assert "RAG" in results[0].text


def test_retriever_returns_requested_top_k():
    chunks = [
        Chunk("LoRA adapts large language models.", "notes.md", None, "c0"),
        Chunk("RAG retrieves external context.", "notes.md", None, "c1"),
        Chunk("Transformers use attention.", "notes.md", None, "c2"),
    ]
    retriever = FaissRetriever(embedder=FakeEmbedder())

    retriever.build_index(chunks)
    results = retriever.retrieve("Tell me about retrieval.", top_k=2)

    assert len(results) == 2
    assert all(result.text for result in results)


def test_more_similar_text_ranks_higher():
    chunks = [
        Chunk("Transformers use self-attention.", "notes.md", None, "transformer"),
        Chunk("LoRA injects low-rank matrices for efficient fine-tuning.", "notes.md", None, "lora"),
        Chunk("RAG combines retrieval with generation.", "notes.md", None, "rag"),
    ]
    retriever = FaissRetriever(embedder=FakeEmbedder())

    retriever.build_index(chunks)
    results = retriever.retrieve("How does LoRA fine-tuning work?", top_k=3)

    assert results[0].chunk_id == "lora"
    assert results[0].score >= results[1].score


def test_retriever_save_and_load_round_trip(tmp_path):
    chunks = [
        Chunk(
            "RAG combines retrieval with generation.",
            "notes.md",
            1,
            "notes_p1_c0",
            {"topic": "rag"},
        )
    ]
    retriever = FaissRetriever(embedder=FakeEmbedder())
    retriever.build_index(chunks)
    retriever.save(str(tmp_path))

    loaded = FaissRetriever(embedder=FakeEmbedder())
    loaded.load(str(tmp_path))
    results = loaded.retrieve("retrieval generation", top_k=1)

    assert results[0].chunk_id == "notes_p1_c0"
    assert results[0].metadata["topic"] == "rag"
