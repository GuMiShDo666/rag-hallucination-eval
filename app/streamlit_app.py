"""Streamlit demo for the RAG hallucination evaluation pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.embedder import Embedder
from src.pipeline import RAGHallucinationPipeline


st.set_page_config(page_title="RAG Hallucination Eval", layout="wide")


def main() -> None:
    """Render the Streamlit application."""

    st.title("RAG Hallucination Eval")

    with st.sidebar:
        docs_path = st.text_input("docs_path", value="data/raw_docs")
        chunk_size = st.slider("chunk_size", min_value=128, max_value=1536, value=512, step=128)
        max_overlap = max(0, chunk_size - 1)
        chunk_overlap = st.slider("chunk_overlap", min_value=0, max_value=max_overlap, value=min(80, max_overlap), step=16)
        top_k = st.slider("top_k", min_value=1, max_value=10, value=5)
        use_query_rewrite = st.checkbox("use_query_rewrite", value=False)
        use_reranker = st.checkbox("use_reranker", value=False)
        use_local_embedding = st.checkbox("local_hashing_embedding", value=True)
        use_mock_llm = st.checkbox("mock_llm", value=True)
        qwen_key = st.text_input("QWEN_API_KEY", value="", type="password")

        if qwen_key:
            os.environ["QWEN_API_KEY"] = qwen_key
            os.environ["LLM_PROVIDER"] = "qwen"
        os.environ["MOCK_LLM"] = "true" if use_mock_llm else "false"

        build_clicked = st.button("Build Index", type="primary")

    if build_clicked:
        _build_pipeline(
            docs_path=docs_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            use_query_rewrite=use_query_rewrite,
            use_reranker=use_reranker,
            use_local_embedding=use_local_embedding,
        )

    question = st.text_input("Question", value="Why is RAG used in question answering?")
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
        if "pipeline" not in st.session_state:
            st.error("Build the index before asking a question.")
            return
        _ask_question(question)


def _build_pipeline(
    docs_path: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    use_query_rewrite: bool,
    use_reranker: bool,
    use_local_embedding: bool,
) -> None:
    embedder = Embedder("hashing") if use_local_embedding else None
    pipeline = RAGHallucinationPipeline(
        docs_path=docs_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        use_query_rewrite=use_query_rewrite,
        use_reranker=use_reranker,
        embedder=embedder,
    )
    try:
        pipeline.build()
    except Exception as exc:
        st.error(f"Index build failed: {exc}")
        return

    st.session_state["pipeline"] = pipeline
    st.success(f"Index built from {len(pipeline.documents)} documents and {len(pipeline.chunks)} chunks.")


def _ask_question(question: str) -> None:
    try:
        result = st.session_state["pipeline"].ask(question)
    except Exception as exc:
        st.error(f"Ask failed: {exc}")
        return

    st.subheader("Answer")
    st.write(result["answer"])

    metric_cols = st.columns(5)
    metric_cols[0].metric("hallucination_rate", _format_metric(result.get("hallucination_rate")))
    metric_cols[1].metric("faithfulness", _format_metric(result.get("faithfulness")))
    metric_cols[2].metric("answer_relevancy", _format_metric(result.get("answer_relevancy")))
    metric_cols[3].metric("context_precision", _format_metric(result.get("context_precision")))
    metric_cols[4].metric("final_judgement", str(result.get("final_judgement")))

    st.subheader("Retrieved Contexts")
    for index, context in enumerate(result["contexts"], start=1):
        with st.expander(f"[{index}] {context['chunk_id']} score={context['score']:.3f}", expanded=index == 1):
            st.write(context["text"])
            st.caption(f"source={context['source']} page={context['page']}")

    st.subheader("Unsupported Spans")
    unsupported_spans = result.get("unsupported_spans", [])
    if not unsupported_spans:
        st.success("No unsupported or contradicted spans detected.")
    for span in unsupported_spans:
        status = str(span.get("status", "unclear")).upper()
        if status == "UNSUPPORTED":
            st.error(f"UNSUPPORTED: {span.get('text', '')}")
        elif status == "CONTRADICTED":
            st.error(f"CONTRADICTED: {span.get('text', '')}")
        else:
            st.warning(f"UNCLEAR: {span.get('text', '')}")
        st.caption(span.get("reason", ""))


def _format_metric(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
