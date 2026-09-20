"""Streamlit chat UI for the Acme multi-agent support assistant."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import GROQ_MODEL, POLICY_DIR, ensure_data_dirs
from src.data_loader import ingest_pdf
from src.graph import ask_sync

STARTERS = [
    "What is the current refund policy?",
    "Give me a quick overview of customer Ema and her past support tickets.",
    "Does Ema's refund ticket match the current refund policy?",
]


def _run_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Routing to an agent..."):
            result = ask_sync(question)
        route = result.get("route") or "unknown"
        st.caption(f"Agent: {route}")
        st.write(result["answer"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "route": route}
    )


def main() -> None:
    st.set_page_config(page_title="Support Assistant", layout="wide")
    ensure_data_dirs()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Support Assistant")
    

    with st.sidebar:
        st.subheader("Settings")
        st.text(f"Model: {GROQ_MODEL}")
        # st.caption("Change GROQ_MODEL in .env and restart Streamlit.")
        uploaded = st.file_uploader("Upload a policy PDF", type=["pdf"])
        if uploaded:
            dest = POLICY_DIR / Path(uploaded.name).name
            dest.write_bytes(uploaded.getvalue())
            with st.spinner("Indexing PDF into FAISS..."):
                info = ingest_pdf(str(dest))
            st.success(f"Ingested {dest.name} ({info.get('chunks', 0)} chunks in folder).")
        st.subheader("Try these")
        for starter in STARTERS:
            if st.button(starter, use_container_width=True):
                _run_question(starter)
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("route"):
                st.caption(f"Agent: {message['route']}")
            st.write(message["content"])

    prompt = st.chat_input("Ask about a policy or a customer...")
    if prompt:
        _run_question(prompt)


if __name__ == "__main__":
    main()
