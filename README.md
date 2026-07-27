# 🔍 RAG Document Q&A System

An end-to-end **Retrieval-Augmented Generation (RAG)** application that lets you upload documents and ask questions about them — powered by **LangChain**, **ChromaDB**, and **Groq's free LLaMA 3 API**.

---

## 🏗️ Architecture

```
Documents (.txt / .pdf)
        ↓
  [Text Splitter]        — RecursiveCharacterTextSplitter (chunk_size=500)
        ↓
  [HuggingFace Embeddings] — all-MiniLM-L6-v2 (runs locally, free)
        ↓
  [ChromaDB Vector Store]  — persistent local vector database
        ↓
  [Retriever]             — top-3 semantic similarity search
        ↓
  [Groq LLM – LLaMA 3]   — free API, generates context-aware answers
        ↓
     Answer + Sources
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-qa-system.git
cd rag-qa-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get your FREE Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up → Create API Key (free, no credit card needed)

### 4. Set your API key
```bash
export GROQ_API_KEY="your_key_here"
```

### 5a. Run via terminal (CLI mode)
```bash
python src/rag_pipeline.py
```

### 5b. Run via Streamlit (Web UI)
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 💡 Demo Use Case — Supply Chain Q&A

A sample document (`docs/supply_chain_overview.txt`) is included covering:
- Yard Management & YMS concepts
- Terminal Throughput optimization
- Demand Forecasting with AI
- RAG in logistics platforms
- GenAI agents in supply chains

Try asking:
> *"What is terminal throughput and how is it improved?"*
> *"How does RAG help in supply chain management?"*
> *"What is yard management?"*

---

## 📁 Project Structure

```
rag-qa-system/
├── app.py                        # Streamlit web UI
├── requirements.txt              # Python dependencies
├── README.md
├── docs/                         # Drop your documents here
│   └── supply_chain_overview.txt # Sample document
├── src/
│   └── rag_pipeline.py           # Core RAG logic
└── notebooks/
    └── rag_walkthrough.ipynb     # Step-by-step explanation
```

---

## 🛠️ Tech Stack

| Component | Tool | Cost |
|---|---|---|
| LLM | Groq – LLaMA 3 8B | ✅ Free |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | ✅ Free |
| Vector DB | ChromaDB (local) | ✅ Free |
| Orchestration | LangChain | ✅ Open source |
| Web UI | Streamlit | ✅ Free |

---

## 🔄 How RAG Works (Simple Explanation)

1. **Ingest** — Load your documents and split into chunks
2. **Embed** — Convert chunks into vector embeddings (numerical representations)
3. **Store** — Save embeddings in ChromaDB for fast similarity search
4. **Retrieve** — When you ask a question, find the top-3 most relevant chunks
5. **Generate** — Send question + retrieved chunks to LLaMA 3 → get a grounded answer

---

## 📌 Extending This Project

- Swap ChromaDB → **Pinecone** for cloud-hosted vector search
- Add **conversation memory** using `ConversationBufferMemory`
- Ingest **CSV/Excel** supply chain data using `CSVLoader`
- Deploy on **Streamlit Cloud** (free hosting)

---

*Built as part of a portfolio project aligned with supply chain AI development.*
