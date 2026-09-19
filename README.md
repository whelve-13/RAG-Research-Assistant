# 📚 RAG Research & Paper Assistant

An end-to-end Retrieval-Augmented Generation (RAG) assistant that indexes academic textbooks and research papers into local vector representations to enable grounded, hallucination-free Q&A with sub-second retrieval latency.

## 🚀 Key Features
- **Document Chunking & Vectorization:** Chunks academic texts using `RecursiveCharacterTextSplitter` and embeds them using local CPU-optimized `sentence-transformers/all-MiniLM-L6-v2` models.
- **Local Vector Indexing:** Persists high-dimensional vector representations in an ephemeral `ChromaDB` index.
- **Low-Latency Synthesis:** Routes retrieved context through Groq Cloud LPUs for conversational completion.
- **Auditable Citations:** Provides verifiable page citations alongside generated answers.

## 🛠️ Tech Stack
- **Frameworks:** LangChain (LCEL), Streamlit
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace Sentence Transformers
- **Inference Engine:** Groq Cloud LPU
- **Language:** Python 3.13

## 💻 Local Setup
1. Clone repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/rag-research-assistant.git
   cd rag-research-assistant