"""End-to-end RAG hallucination evaluation pipeline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.chunker import split_documents
from src.config import Settings, load_settings
from src.document_loader import load_documents
from src.embedder import Embedder
from src.evaluator import RAGEvaluator
from src.generator import LLMGenerator
from src.hallucination_detector import HallucinationDetector
from src.retriever import FaissRetriever, RetrievedChunk


class RAGHallucinationPipeline:
    """Main entry point for document indexing, retrieval, generation, and evaluation."""

    def __init__(
        self,
        docs_path: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
        use_query_rewrite: bool = False,
        use_reranker: bool = False,
        settings: Settings | None = None,
        embedder: Any | None = None,
        generator: LLMGenerator | None = None,
        detector: HallucinationDetector | None = None,
        evaluator: RAGEvaluator | None = None,
    ):
        """Create a pipeline. Optional components are useful for tests and experiments."""

        self.settings = settings or load_settings()
        self.docs_path = docs_path
        self.chunk_size = chunk_size or self.settings.default_chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else self.settings.default_chunk_overlap
        self.top_k = top_k or self.settings.default_top_k
        self.use_query_rewrite = use_query_rewrite
        self.use_reranker = use_reranker

        self.embedder = embedder
        mock_llm = self.settings.mock_llm or not self._has_provider_key(self.settings.llm_provider)
        self.generator = generator or LLMGenerator(
            provider=self.settings.llm_provider,
            model=self.settings.active_model,
            mock_mode=mock_llm,
        )
        self.detector = detector or HallucinationDetector(provider="qwen", use_llm_judge=False)
        self.evaluator = evaluator or RAGEvaluator(generator=self.generator, detector=self.detector)

        self.retriever: FaissRetriever | None = None
        self.documents = []
        self.chunks = []
        self.is_built = False

    def build(self) -> None:
        """Load documents, split them into chunks, and build the vector index."""

        self.documents = load_documents(self.docs_path)
        if not self.documents:
            raise ValueError(f"No supported non-empty documents were loaded from {self.docs_path}.")

        self.chunks = split_documents(
            self.documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not self.chunks:
            raise ValueError("Documents were loaded but no non-empty chunks were created.")

        if self.embedder is None:
            self.embedder = Embedder(self.settings.embedding_model)
        self.retriever = FaissRetriever(self.embedder)
        self.retriever.build_index(self.chunks)
        self.is_built = True

    def ask(self, question: str) -> dict:
        """Answer a question and return contexts, hallucination spans, and metrics."""

        if not self.is_built or self.retriever is None:
            raise ValueError("Pipeline has not been built. Call build() before ask().")
        if not question.strip():
            raise ValueError("Question must not be empty.")

        query = self._rewrite_query(question) if self.use_query_rewrite else question
        contexts = self.retriever.retrieve(query, top_k=self.top_k)
        contexts = self._rerank(query, contexts) if self.use_reranker else contexts

        generation = self.generator.generate(question, contexts)
        context_texts = [context.text for context in contexts]
        hallucination = self.detector.detect(question, contexts, generation.answer)
        evaluation = self.evaluator.evaluate_single(
            question=question,
            answer=generation.answer,
            contexts=context_texts,
        )

        return {
            "question": question,
            "answer": generation.answer,
            "contexts": [_retrieved_chunk_to_dict(context) for context in contexts],
            "unsupported_spans": [
                asdict(span)
                for span in hallucination.spans
                if span.status in {"unsupported", "contradicted"}
            ],
            "hallucination_rate": hallucination.hallucination_rate,
            "faithfulness": evaluation.faithfulness,
            "answer_relevancy": evaluation.answer_relevancy,
            "context_precision": evaluation.context_precision,
            "citation_accuracy": evaluation.citation_accuracy,
            "final_judgement": hallucination.final_judgement,
        }

    def _rewrite_query(self, question: str) -> str:
        return question

    def _rerank(self, query: str, contexts: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return contexts

    def _has_provider_key(self, provider: str) -> bool:
        if provider == "qwen":
            return bool(self.settings.qwen_api_key)
        if provider == "deepseek":
            return bool(self.settings.deepseek_api_key)
        if provider == "openai":
            return bool(self.settings.openai_api_key)
        return False


def _retrieved_chunk_to_dict(chunk: RetrievedChunk) -> dict:
    return {
        "text": chunk.text,
        "score": chunk.score,
        "source": chunk.source,
        "page": chunk.page,
        "chunk_id": chunk.chunk_id,
        "metadata": dict(chunk.metadata or {}),
    }
