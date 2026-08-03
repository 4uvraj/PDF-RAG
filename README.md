# 📄 Secure PDF RAG

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload a PDF and ask questions about its contents — powered by **LangChain**, **FAISS** vector search, and **OpenAI** embeddings/chat models, with a built-in **daily token budget guard** to keep API costs under control.

---

## ✨ Features

- 📤 **PDF upload & ingestion** — parses PDFs directly in-browser via Streamlit.
- ✂️ **Smart chunking** — splits documents using a recursive character splitter (800-token chunks, 150-token overlap) for better retrieval quality.
- 🧠 **Semantic search with FAISS** — embeds chunks with OpenAI embeddings and retrieves the most relevant context using **MMR (Maximal Marginal Relevance)** search.
- 💬 **Conversational memory** — keeps the last 8 turns of chat history so follow-up questions stay contextual.
- 🛡️ **Grounded answers** — the LLM is explicitly prompted to answer only from retrieved context and say "I don't know" when the answer isn't in the document, reducing hallucination.
- 📊 **Daily token budget system** — tracks cumulative token usage per day (`usage.json`), reserving separate quotas for embeddings vs. chat, and blocks requests once the daily limit is hit — a lightweight cost-control mechanism for a public-facing LLM app.
- ⚡ **Cached vector store** — uses `st.cache_resource` so re-querying the same document doesn't re-embed it.

---

## 🏗️ How It Works

```
 PDF Upload
     │
     ▼
 PyPDFLoader (extract text)
     │
     ▼
 RecursiveCharacterTextSplitter (chunking)
     │
     ▼
 OpenAI Embeddings ──▶ FAISS Vector Store
     │
     ▼
 User Question
     │
     ▼
 MMR Retriever (top-k relevant chunks)
     │
     ▼
 Prompt Template (history + context + question)
     │
     ▼
 ChatOpenAI (gpt-4o-mini) ──▶ Answer
     │
     ▼
 Token usage logged to usage.json
```

Every step — upload, embedding, and chat — checks the remaining daily token budget first, so the app degrades gracefully with a clear error message instead of racking up unexpected API costs.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Vector Store | FAISS |
| Embeddings | OpenAI Embeddings |
| LLM | OpenAI `gpt-4o-mini` (via `ChatOpenAI`) |
| PDF Parsing | `PyPDFLoader` (langchain-community) |
| Config | python-dotenv |
| Usage Tracking | Local JSON file (`usage.json`) |

---

## 📂 Project Structure

```
PDF-RAG/
├── .streamlit/           # Streamlit configuration
├── app.py                 # Main RAG application (single-file)
├── requirements.txt        # Python dependencies
├── runtime.txt              # Python runtime version (for deployment)
├── usage.json                # Daily token usage log (auto-generated/updated)
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone the repository
```bash
git clone https://github.com/4uvraj/PDF-RAG.git
cd PDF-RAG
```

### 2. Create a virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

### 5. Use it
1. Upload a PDF (max 5 MB).
2. Wait for it to be chunked and embedded into FAISS.
3. Ask a question in natural language — the app retrieves relevant chunks and answers using the document as ground truth.

---

## ⚙️ Configuration

Token budgets are defined at the top of `app.py` and can be tuned:

```python
DAILY_TOKEN_LIMIT = 100000     # total tokens allowed per day
EMBEDDING_RESERVE = 60000      # max tokens reserved for embedding new documents
CHAT_RESERVE = 40000           # max tokens reserved for chat responses
```

Usage resets automatically at the start of a new day and is persisted in `usage.json`.

---

## 🧠 Design Decisions & Trade-offs

- **MMR over plain similarity search** — reduces redundant chunks in the retrieved context, giving the LLM more diverse, relevant information instead of near-duplicate passages.
- **Token estimation via character count (`len(text) / 4`)** — a fast, dependency-free approximation instead of a full tokenizer call, trading precision for speed and simplicity.
- **File-based usage tracking** — simple and dependency-free for a single-instance deployment; would need to move to a shared database (e.g., Redis/Postgres) to scale across multiple app instances.
- **Explicit "I don't know" instruction in the prompt** — prioritizes factual grounding over fluency, reducing hallucinated answers at the cost of occasionally declining to answer.

---

## 🗺️ Roadmap

- [ ] Support multi-PDF sessions and cross-document Q&A
- [ ] Move usage tracking to a proper database for multi-user/multi-instance support
- [ ] Add source citations (page numbers) alongside answers
- [ ] Swap character-count token estimation for a real tokenizer (`tiktoken`)
- [ ] Add automated tests for chunking and retrieval logic
- [ ] Dockerize for one-command setup

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/4uvraj/PDF-RAG/issues).

## 📄 License

This project currently has no license specified. Consider adding an [MIT License](https://choosealicense.com/licenses/mit/) if you intend for others to reuse this code.

## 🙋 Author

**4uvraj** — [GitHub Profile](https://github.com/4uvraj)
