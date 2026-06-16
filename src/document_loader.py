"""Document loading utilities for text, Markdown, and PDF inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings


@dataclass
class Document:
    """A normalized document page or text file loaded from disk."""

    text: str
    source: str
    page: int | None = None
    metadata: dict = field(default_factory=dict)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(path: str) -> list[Document]:
    """Load supported documents from a file or directory path.

    Text and Markdown files are loaded as one document each. PDF files are
    loaded one page per document. Empty files and unsupported extensions are
    skipped with warnings.
    """

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Document path does not exist: {path}")

    files = _iter_supported_files(input_path)
    documents: list[Document] = []
    for file_path in files:
        if file_path.suffix.lower() in {".txt", ".md"}:
            documents.extend(_load_text_file(file_path))
        elif file_path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf_file(file_path))

    return documents


def _iter_supported_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            warnings.warn(f"Skipping unsupported file type: {path}", stacklevel=2)
            return []
        return [path]

    files: list[Path] = []
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name == ".gitkeep":
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            warnings.warn(f"Skipping unsupported file type: {file_path}", stacklevel=2)
            continue
        files.append(file_path)
    return files


def _load_text_file(path: Path) -> list[Document]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        warnings.warn(f"Skipping empty document: {path}", stacklevel=2)
        return []

    return [
        Document(
            text=text,
            source=str(path),
            page=None,
            metadata={"file_name": path.name, "extension": path.suffix.lower()},
        )
    ]


def _load_pdf_file(path: Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError:
        warnings.warn("pypdf is not installed; skipping PDF file: {path}", stacklevel=2)
        return []

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        warnings.warn(f"Could not read PDF {path}: {exc}", stacklevel=2)
        return []

    documents: list[Document] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            warnings.warn(f"Could not extract page {page_index} from {path}: {exc}", stacklevel=2)
            continue
        if not text:
            warnings.warn(f"Skipping empty PDF page {page_index}: {path}", stacklevel=2)
            continue
        documents.append(
            Document(
                text=text,
                source=str(path),
                page=page_index,
                metadata={
                    "file_name": path.name,
                    "extension": ".pdf",
                    "page": page_index,
                },
            )
        )
    return documents
