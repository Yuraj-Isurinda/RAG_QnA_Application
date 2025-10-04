import os, time, random, uuid
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document


from rag_store import build_store

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY, "GOOGLE_API_KEY is not set"

store = build_store(GOOGLE_API_KEY)

# 1) LLM and embeddings (use flash for quota headroom)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    max_output_tokens=512,
    google_api_key=GOOGLE_API_KEY,
)

def format_docs(docs: list[Document]) -> str:
    MAX_CHARS = 2000
    return "\n\n---\n\n".join(
        f"[p{d.metadata.get('page','?')} {d.metadata.get('source','?')}] {d.page_content}"
        for d in docs
    )[:MAX_CHARS]

def format_chat_history(history: list[str]) -> str:
    out = []
    for human, ai in zip(history[::2], history[1::2]):
        out.append(f"Human: {human}\nAI: {ai}")
    return "\n".join(out)

template = """You are a helpful AI assistant. Answer the question based on the context below and the conversation history. If you don't know the answer, say you don't know.

Context:
{context}

Conversation History:
{history}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)


def get_context(question: str) -> str:
    retriever = store.retriever(k=4)
    docs = retriever.invoke(question)
    return format_docs(docs)


# 6) Chain
chain = (
    {
        "context": RunnableLambda(lambda x: get_context(x["question"])),
        "question": RunnableLambda(lambda x: x["question"]),
        "history": RunnableLambda(lambda x: format_chat_history(x.get("history", []))),
    }
    | prompt
    | llm
    | StrOutputParser()
)
rag_store = store