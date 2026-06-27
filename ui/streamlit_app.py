import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vectorstore import get_or_create_vectorstore
from app.graph import build_graph

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 DocMind AI")
st.caption("Agentic Document Intelligence — powered by LangGraph")

# ============================================================
# SESSION STATE INIT
# ============================================================

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# ============================================================
# SIDEBAR — PDF UPLOAD
# ============================================================

with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:
        # Save uploaded file to disk
        pdf_path = f"/tmp/uploaded_{uploaded_file.name}"

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process only if new file
        if st.session_state.pdf_path != pdf_path:
            with st.spinner("Processing PDF..."):
                st.session_state.vectorstore = get_or_create_vectorstore(pdf_path)
                st.session_state.pdf_path = pdf_path
                st.session_state.pdf_processed = True
                st.session_state.chat_history = []  # reset chat

            st.success(f"✅ {uploaded_file.name} processed!")

    if st.session_state.pdf_processed:
        st.info(f"📄 Active: {st.session_state.pdf_path}")

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ============================================================
# MAIN — CHAT INTERFACE
# ============================================================

if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF from the sidebar to get started")

else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Show sources if available
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**Page {source['page']}** — `{source['source']}`")
                        st.caption(source["snippet"])

    # Chat input
    question = st.chat_input("Ask a question about your document...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)

        # Add to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("🧠 Agent thinking..."):
                result = st.session_state.graph.invoke({
                    "question": question,
                    "pdf_path": st.session_state.pdf_path,
                    "documents": [],
                    "filtered_documents": [],
                    "answer": "",
                    "sources": [],
                    "retry_count": 0,
                    "hallucination_status": ""
                })

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            # Handle empty answer (max retries hit)
            if not answer or answer.strip() == "" or answer == "not_related":
                answer = f"❌ Couldn't find relevant information about **'{question}'** in the document. Try rephrasing or check if this topic is covered in your PDF."
            st.write(answer)

            # Show sources
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.markdown(f"**Page {source['page']}** — `{source['source']}`")
                        st.caption(source["snippet"])

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })