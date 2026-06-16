"""Vector retrieval backed by FAISS with a numpy fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np

from src.chunker import Chunk
from src.embedder import Embedder

try:
    import faiss
except ImportError:  # pragma: no cover - exercised indirectly when FAISS is absent.
    faiss = None


@dataclass
class RetrievedChunk:
    """A retrieved chunk and its similarity score."""

    text: str
    score: float
    source: str
    page: int | None
    chunk_id: str
    metadata: dict = field(default_factory=dict)


class FaissRetriever:
    """Build, persist, and query a vector index for chunks."""

    def __init__(self, embedder: Embedder):
        """Create a retriever using the supplied embedder."""

        self.embedder = embedder
        self.index = None
        self.embeddings: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self.dimension: int | None = None

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build a vector index from chunks."""

        if not chunks:
            raise ValueError("Cannot build an index from an empty chunk list.")

        texts = [chunk.text for chunk in chunks]
        embeddings = _normalize(np.asarray(self.embedder.encode(texts), dtype=np.float32))
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        self.chunks = list(chunks)
        self.embeddings = embeddings
        self.dimension = embeddings.shape[1]

        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
        else:
            self.index = None

    def save(self, path: str) -> None:
        """Save the index and chunk metadata to a directory."""

        if self.embeddings is None or self.dimension is None:
            raise ValueError("No index has been built.")

        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(output_dir / "index.faiss"))

        payload = {
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "dimension": self.dimension,
        }
        with (output_dir / "metadata.pkl").open("wb") as file:
            pickle.dump(payload, file)

    def load(self, path: str) -> None:
        """Load a saved index from a directory."""

        input_dir = Path(path)
        with (input_dir / "metadata.pkl").open("rb") as file:
            payload = pickle.load(file)

        self.chunks = payload["chunks"]
        self.embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        self.dimension = int(payload["dimension"])

        faiss_path = input_dir / "index.faiss"
        if faiss is not None and faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
        else:
            self.index = None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve the top-k chunks most similar to the query."""

        if self.embeddings is None or not self.chunks:
            raise ValueError("No index has been built or loaded.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_embedding = _normalize(np.asarray(self.embedder.encode([query]), dtype=np.float32))
        limit = min(top_k, len(self.chunks))

        if faiss is not None and self.index is not None:
            scores, indices = self.index.search(query_embedding, limit)
            score_list = scores[0]
            index_list = indices[0]
        else:
            similarities = self.embeddings @ query_embedding[0]
            index_list = np.argsort(-similarities)[:limit]
            score_list = similarities[index_list]

        results: list[RetrievedChunk] = []
        for score, chunk_index in zip(score_list, index_list, strict=False):
            if chunk_index < 0:
                continue
            chunk = self.chunks[int(chunk_index)]
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    score=float(score),
                    source=chunk.source,
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    metadata=dict(chunk.metadata or {}),
                )
            )
        return results


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embeddings / norms).astype(np.float32)
