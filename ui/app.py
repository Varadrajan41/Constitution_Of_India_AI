"""Streamlit chat UI for the Constitution of India RAG.

Run from the project root:
    export PYTHONPATH=.
    streamlit run ui/app.py
"""
import streamlit as st

from backend import config
from backend import query as rag

st.set_page_config(page_title="Constitution of India - RAG", page_icon="book")
st.title("Constitution of India - RAG")
st.caption("Hybrid retrieval + reranker, answered by a local legal LLM (Ollama).")

with st.sidebar:
    st.header("Retrieval")
    hybrid = st.toggle("Hybrid (both collections)", value=config.HYBRID,
                       help="Pull from naive + structured, then rerank.")
    rerank_on = st.toggle("Rerank (cross-encoder)", value=config.RERANK_ENABLED,
                          help=config.RERANKER_MODEL)
    config.RERANK_ENABLED = rerank_on
    strategy = st.radio(
        "Single-collection strategy (when hybrid is off)",
        options=["structured", "naive"],
        index=0 if config.DEFAULT_STRATEGY == "structured" else 1,
        disabled=hybrid,
    )
    st.caption(f"Embed: `{config.EMBEDDING_MODEL}`")
    st.caption(f"LLM: `{config.OLLAMA_MODEL}`")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask about an Article, right, or provision...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, reasoning..."):
            try:
                text, passages = rag.answer(question, hybrid=hybrid, strategy=strategy)
            except Exception as exc:  # surface setup issues in the UI
                text, passages = f"**Error:** {exc}", []
        st.markdown(text)
        if passages:
            arts = sorted({m.get("article") for _, m, _ in passages if m.get("article")})
            if arts:
                st.caption("Sources: Articles " + ", ".join(arts))
            with st.expander("Retrieved passages"):
                for doc, meta, score in passages:
                    tag = meta.get("article") or meta.get("schedule") or meta.get("type")
                    st.markdown(
                        f"**{meta.get('strategy','?')} · {tag} · score {score:.3f}**"
                    )
                    st.text(doc[:600].strip())
    st.session_state.history.append({"role": "assistant", "content": text})
