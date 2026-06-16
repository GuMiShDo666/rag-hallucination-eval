from src.chunker import split_documents
from src.document_loader import Document, load_documents


def test_load_documents_reads_text_and_markdown(tmp_path):
    txt_path = tmp_path / "note.txt"
    md_path = tmp_path / "guide.md"
    unsupported_path = tmp_path / "image.png"

    txt_path.write_text("Plain text document.", encoding="utf-8")
    md_path.write_text("# Markdown\n\nRAG grounds answers.", encoding="utf-8")
    unsupported_path.write_text("not loaded", encoding="utf-8")

    documents = load_documents(str(tmp_path))

    assert len(documents) == 2
    assert {doc.metadata["extension"] for doc in documents} == {".txt", ".md"}
    assert all(doc.text for doc in documents)


def test_split_documents_creates_non_empty_chunks():
    document = Document(text="abcdefghij", source="sample.txt", metadata={"topic": "letters"})

    chunks = split_documents([document], chunk_size=4, chunk_overlap=1)

    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij"]
    assert all(chunk.text for chunk in chunks)


def test_chunk_overlap_is_applied():
    document = Document(text="abcdefghijklmnopqrstuvwxyz", source="alphabet.txt")

    chunks = split_documents([document], chunk_size=10, chunk_overlap=3)

    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert chunks[1].text[-3:] == chunks[2].text[:3]


def test_metadata_and_chunk_id_are_preserved():
    document = Document(
        text="RAG combines retrieval with generation.",
        source="/tmp/sample_llm_notes.md",
        page=2,
        metadata={"section": "rag"},
    )

    chunks = split_documents([document], chunk_size=100, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].source == document.source
    assert chunks[0].page == 2
    assert chunks[0].chunk_id == "sample_llm_notes.md_p2_c0"
    assert chunks[0].metadata["section"] == "rag"
    assert chunks[0].metadata["chunk_index"] == 0


def test_empty_documents_do_not_create_chunks():
    document = Document(text="   ", source="empty.txt")

    chunks = split_documents([document], chunk_size=10, chunk_overlap=2)

    assert chunks == []
