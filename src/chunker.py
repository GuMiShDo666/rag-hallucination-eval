"""Text chunking utilities for loaded documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.document_loader import Document


@dataclass
class Chunk:
    """A chunk of text with source metadata preserved from its document."""

    text: str
    source: str
    page: int | None
    chunk_id: str
    metadata: dict = field(default_factory=dict)


def split_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """Split documents into non-empty overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(_split_text(document.text, chunk_size, chunk_overlap)):
            chunk_id = _make_chunk_id(document.source, document.page, index)
            metadata = dict(document.metadata or {})
            metadata.update(
                {
                    "source": document.source,
                    "page": document.page,
                    "chunk_index": index,
                }
            )
            chunks.append(
                Chunk(
                    text=text,
                    source=document.source,
                    page=document.page,
                    chunk_id=chunk_id,
                    metadata=metadata,
                )
            )
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    clean_text = text.strip()
    if not clean_text:
        return

    start = 0
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunk_text = clean_text[start:end].strip()
        if chunk_text:
            yield chunk_text
        if end == len(clean_text):
            break
        start = end - chunk_overlap


def _make_chunk_id(source: str, page: int | None, index: int) -> str:
    file_name = Path(source).name or "document"
    page_part = f"p{page}" if page is not None else "pnone"
    return f"{file_name}_{page_part}_c{index}"
