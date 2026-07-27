"""
RAG Pipeline — LangChain + ChromaDB + Groq (LLaMA 3)
Loads documents → chunks → embeds → stores in ChromaDB → retrieves → answers via LLM
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


# ── Configuration ────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
CHROMA_DB_DIR  = "./chroma_db"
DOCS_DIR       = "./docs"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"   # free, runs locally
LLM_MODEL      = "llama3-8b-8192"                           # free on Groq


# ── Step 1: Load Documents ────────────────────────────────────────────────────
def load_documents(docs_dir: str = DOCS_DIR):
    """Load .txt and .pdf files from the docs/ folder."""
    loaders = [
        DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]
    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"[WARN] Loader skipped: {e}")
    print(f"[INFO] Loaded {len(documents)} document(s).")
    return documents


# ── Step 2: Chunk Documents ───────────────────────────────────────────────────
def chunk_documents(documents):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Created {len(chunks)} chunks.")
    return chunks


# ── Step 3: Embed + Store in ChromaDB ────────────────────────────────────────
def build_vectorstore(chunks, persist_dir: str = CHROMA_DB_DIR):
    """Embed chunks using HuggingFace and store in ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"[INFO] Vector store saved to '{persist_dir}'.")
    return vectorstore


# ── Step 4: Load Existing ChromaDB ───────────────────────────────────────────
def load_vectorstore(persist_dir: str = CHROMA_DB_DIR):
    """Load a previously built ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectorstore


# ── Step 5: Build QA Chain ───────────────────────────────────────────────────
def build_qa_chain(vectorstore):
    """Connect ChromaDB retriever to Groq LLM via LangChain RetrievalQA."""
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2
    )

    prompt_template = """You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain


# ── Step 6: Query ─────────────────────────────────────────────────────────────
def ask(qa_chain, question: str):
    """Run a question through the RAG pipeline."""
    result = qa_chain.invoke({"query": question})
    print(f"\n🔍 Question: {question}")
    print(f"💬 Answer:   {result['result']}")
    print(f"\n📄 Sources:")
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        print(f"   - {src}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Build or load vector store
    if not os.path.exists(CHROMA_DB_DIR):
        print("[INFO] Building vector store for the first time...")
        docs   = load_documents()
        chunks = chunk_documents(docs)
        vs     = build_vectorstore(chunks)
    else:
        print("[INFO] Loading existing vector store...")
        vs = load_vectorstore()

    qa = build_qa_chain(vs)

    # Interactive Q&A loop
    print("\n=== RAG Document Q&A System ===")
    print("Type your question (or 'quit' to exit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            ask(qa, q)
