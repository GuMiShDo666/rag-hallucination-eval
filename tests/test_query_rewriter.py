from src.query_rewriter import QueryRewriter, _clean_rewritten_query


def test_query_rewriter_mock_returns_original_query():
    rewriter = QueryRewriter(provider="qwen", model="qwen-plus", mock_mode=True)

    result = rewriter.rewrite("Why is RAG used in question answering?")

    assert result.original_query == "Why is RAG used in question answering?"
    assert result.rewritten_query == "Why is RAG used in question answering?"
    assert result.changed is False
    assert result.model == "qwen-plus"


def test_query_rewriter_without_api_key_falls_back_to_original(monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    rewriter = QueryRewriter(provider="qwen", model="qwen-plus", mock_mode=False)

    result = rewriter.rewrite("How does LoRA reduce trainable parameters?")

    assert result.rewritten_query == "How does LoRA reduce trainable parameters?"
    assert result.changed is False


def test_clean_rewritten_query_removes_common_prefixes():
    assert _clean_rewritten_query("Search query: LoRA low-rank adapter fine-tuning") == (
        "LoRA low-rank adapter fine-tuning"
    )
    assert _clean_rewritten_query('"RAG retrieval grounded question answering"') == (
        "RAG retrieval grounded question answering"
    )
