"""Streamlit chat UI for the Constitution of India RAG.

Run from the project root:
    export PYTHONPATH=.
    streamlit run ui/app.py
"""
import streamlit as st

from backend import config
from backend import query as rag

EXAMPLES = [
    "What does Article 21 guarantee?",
    "How is the President of India elected?",
    "List the fundamental duties of citizens.",
    "Can fundamental rights enforcement be suspended during an emergency?",
]

st.set_page_config(page_title="Constitution of India - RAG", page_icon="📖")
st.title("Constitution of India - RAG")
st.caption("Hybrid retrieval + reranker, answered by a local LLM (Ollama).")


def show_rewritten(original: str, search_query: str):
    """Show the standalone retrieval query when rewrite changed the user message."""
    if rag.query_was_rewritten(original, search_query):
        st.caption(f"Rewritten for retrieval: _{search_query}_")


def render_answer(text, passages, search_query=None, user_question=None):
    """Render an assistant answer with fallback warning, sources, and passages."""
    if user_question and search_query:
        show_rewritten(user_question, search_query)
    if "general knowledge (not in retrieved text" in text.lower():
        st.warning(
            "This answer includes general knowledge not found in the retrieved "
            "text — verify independently."
        )
    if not passages:
        return
    arts = sorted({m.get("article") for _, m, _ in passages if m.get("article")})
    if arts:
        st.caption("Sources: Articles " + ", ".join(arts))
    with st.expander("Retrieved passages"):
        for doc, meta, score in passages:
            tag = meta.get("article") or meta.get("schedule") or meta.get("type")
            st.markdown(f"**{meta.get('strategy', '?')} · {tag} · score {score:.3f}**")
            st.text(doc[:600].strip())


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
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for i, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            user_q = None
            if i > 0 and st.session_state.history[i - 1]["role"] == "user":
                user_q = st.session_state.history[i - 1]["content"]
            render_answer(
                turn["content"],
                turn.get("passages", []),
                search_query=turn.get("search_query"),
                user_question=user_q,
            )

# Starter prompts (only on an empty chat).
pending = None
if not st.session_state.history:
    st.markdown("**Try an example:**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            pending = ex

question = st.chat_input("Ask about an Article, right, or provision...") or pending
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        prior = st.session_state.history[:-1]  # history excluding the new question
        try:
            with st.spinner("Retrieving, reranking..."):
                stream, passages, search_query = rag.answer_stream(
                    question, hybrid=hybrid, strategy=strategy, history=prior
                )
            text = st.write_stream(stream)
        except Exception as exc:  # surface setup issues in the UI
            text, passages, search_query = f"**Error:** {exc}", [], question
            st.markdown(text)
        render_answer(text, passages, search_query, question)
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": text,
            "passages": passages,
            "search_query": search_query,
        }
    )
