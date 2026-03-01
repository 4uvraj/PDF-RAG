import os
import json
import datetime
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================
# CONFIG
# =========================
DAILY_TOKEN_LIMIT = 100000
EMBEDDING_RESERVE = 60000   # reserve 60k for embeddings
CHAT_RESERVE = 40000        # reserve 40k for chat
USAGE_FILE = "usage.json"


# =========================
# LOAD ENV
# =========================
load_dotenv()


# =========================
# TOKEN TRACKING SYSTEM
# =========================
def load_usage():
    if not os.path.exists(USAGE_FILE):
        return {"date": "", "total_tokens": 0}

    with open(USAGE_FILE, "r") as f:
        return json.load(f)


def save_usage(data):
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f)


def update_token_usage(tokens_used):
    usage = load_usage()
    today = str(datetime.date.today())

    if usage["date"] != today:
        usage = {"date": today, "total_tokens": 0}

    usage["total_tokens"] += tokens_used
    save_usage(usage)


def get_remaining_tokens():
    usage = load_usage()
    today = str(datetime.date.today())

    if usage["date"] != today:
        return DAILY_TOKEN_LIMIT

    return DAILY_TOKEN_LIMIT - usage["total_tokens"]


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Secure PDF RAG")

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = []

st.title("Secure PDF RAG")

uploaded_file = st.file_uploader("Upload PDF (Max 5MB)", type="pdf")

if uploaded_file:

    # Check remaining budget before anything
    remaining_tokens = get_remaining_tokens()

    if remaining_tokens <= 0:
        st.error("Daily token limit reached. Try tomorrow.")
        st.stop()

    # Save file
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load full text to estimate tokens
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    full_text = " ".join([doc.page_content for doc in documents])

    estimated_tokens = int(len(full_text) / 4)

    # Protect embedding budget
    if estimated_tokens > EMBEDDING_RESERVE:
        st.error("Document too large for embedding quota today.")
        st.stop()

    if estimated_tokens > remaining_tokens:
        st.error("Not enough remaining daily budget for this file.")
        st.stop()

    @st.cache_resource
    def create_vectorstore(docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        return FAISS.from_documents(chunks, embeddings)

    vectorstore = create_vectorstore(documents)

    # After embedding, update token usage
    update_token_usage(estimated_tokens)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=300
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Use retrieved context for factual answers.
Use conversation history only for continuity.
Answer in under 150 words.
If answer not found, say "I don't know".

Conversation History:
{history}

Retrieved Context:
{context}

Question:
{question}
""")

    rag_chain = (
        {
            "history": lambda x: "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in st.session_state.get("memory", [])[-8:]
            ),
            "context": retriever,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    query = st.text_input("Ask a question")

    if query:

        remaining_tokens = get_remaining_tokens()
        if remaining_tokens <= 0:
            st.error("Daily token limit reached.")
            st.stop()

        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)

        st.subheader("Answer:")
        st.write(response)

        # Estimate chat cost (rough approx)
        estimated_chat_tokens = int((len(query) + len(response)) / 4)
        update_token_usage(estimated_chat_tokens)

        # Update memory
        st.session_state.memory.append(
            {"role": "user", "content": query}
        )
        st.session_state.memory.append(
            {"role": "assistant", "content": response}
        )

        st.session_state.memory = st.session_state.memory[-8:]
