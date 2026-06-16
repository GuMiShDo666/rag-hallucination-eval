"""Embedding model wrapper with sentence-transformers and local fallback."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence
import warnings

import numpy as np


class Embedder:
    """Encode text with sentence-transformers or a deterministic local fallback."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", fallback_dimension: int = 384):
        """Load an embedding model by name.

        If the model cannot be loaded, for example because model weights cannot
        be downloaded, a deterministic hashing embedder is used so experiments
        still run end to end.
        """

        self.model_name = model_name
        self.fallback_dimension = fallback_dimension
        self.model = None
        if model_name.lower() in {"hashing", "local-hashing", "fallback"}:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            warnings.warn(
                "sentence-transformers is not installed; using deterministic hashing embeddings.",
                stacklevel=2,
            )
            return

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            warnings.warn(
                f"Could not load embedding model '{model_name}': {exc}. "
                "Using deterministic hashing embeddings.",
                stacklevel=2,
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return normalized float32 embeddings for one or more texts."""

        if isinstance(texts, str):
            texts = [texts]
        if self.model is None:
            return self._encode_with_hashing(list(texts))
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _encode_with_hashing(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.fallback_dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "little") % self.fallback_dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row_index, column] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)
