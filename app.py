"""
Streamlit UI for RAG Document Q&A System
Run: streamlit run app.py
"""

import os
import streamlit as st
from src.rag_pipeline import (
    load_documents, chunk_documents,
    build_vectorstore, load_vectorstore, build_qa_chain
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Document Q&A System")
st.caption("Powered by LangChain · ChromaDB · Groq (LLaMA 3) · HuggingFace Embeddings")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password",
                              help="Get your free key at console.groq.com")
    st.markdown("---")
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        os.makedirs("./docs", exist_ok=True)

        # Save uploaded files to docs/
        for f in uploaded_files:
            with open(f"./docs/{f.name}", "wb") as out:
                out.write(f.read())

        if st.button("🚀 Build Knowledge Base"):
            with st.spinner("Loading and chunking documents..."):
                docs   = load_documents()
                chunks = chunk_documents(docs)
            with st.spinner("Embedding and storing in ChromaDB..."):
                build_vectorstore(chunks)
            st.success(f"✅ Knowledge base built from {len(docs)} document(s)!")
            st.session_state["kb_ready"] = True

    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("- 🦜 LangChain")
    st.markdown("- 🗄️ ChromaDB")
    st.markdown("- ⚡ Groq (LLaMA 3 – free)")
    st.markdown("- 🤗 HuggingFace Embeddings")

# ── Main Chat Interface ───────────────────────────────────────────────────────
CHROMA_DB_DIR = "./chroma_db"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Load QA chain if KB exists
if os.path.exists(CHROMA_DB_DIR) and groq_key:
    os.environ["GROQ_API_KEY"] = groq_key
    if st.session_state.qa_chain is None:
        with st.spinner("Loading knowledge base..."):
            vs = load_vectorstore()
            st.session_state.qa_chain = build_qa_chain(vs)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload documents and build the knowledge base first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": prompt})
                answer = result["result"]
                sources = result.get("source_documents", [])

            st.markdown(answer)

            # Show sources
            if sources:
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(sources, 1):
                        src = doc.metadata.get("source", "Unknown")
                        st.markdown(f"**{i}.** `{src}`")
                        st.markdown(f"> {doc.page_content[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
