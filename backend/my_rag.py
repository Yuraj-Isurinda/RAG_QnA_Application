import os, time, random, uuid
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY, "GOOGLE_API_KEY is not set"
PDF_PATH = os.getenv("PDF_PATH")
assert PDF_PATH, "PDF_PATH is not set"

# 1) LLM and embeddings (use flash for quota headroom)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.1,
    max_output_tokens=512,
    google_api_key=GOOGLE_API_KEY,
)
emb = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GOOGLE_API_KEY)

# 2) Load + split docs
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
assert docs, f"No pages loaded from {PDF_PATH}"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
assert splits, "No chunks produced"

# 3) Chroma (persistent)
vector_store = Chroma(
    collection_name="my_collection",
    persist_directory="./chroma_db",
    embedding_function=emb,
)

# 4) Add texts (batched + basic retry on 429)
def batched(seq, n): 
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def backoff(retry): 
    time.sleep(min(60, (2 ** retry) + random.random()))

batch_size = 32
for batch in batched(splits, batch_size):
    ids = [getattr(d, "id", None) or str(uuid.uuid4()) for d in batch]
    texts = [d.page_content for d in batch]
    metas = [d.metadata for d in batch]
    retry = 0
    while True:
        try:
            vector_store.add_texts(texts=texts, metadatas=metas, ids=ids)
            break
        except Exception as e:
            s = str(e).lower()
            if "429" in s or "quota" in s:
                backoff(retry); retry += 1; continue
            raise

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 5) Prompt + formatting
def format_docs(docs: list[Document]) -> str:
    MAX_CHARS = 2000  # keep input tokens down
    text = "\n\n---\n\n".join(f"[p{d.metadata.get('page', '?')}] {d.page_content}" for d in docs)
    return text[:MAX_CHARS]

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

# 6) Chain
chain = (
    {
        "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"]))),
        "question": RunnablePassthrough(),
        "history": RunnableLambda(lambda x: format_chat_history(x.get("history", []))),
    }
    | prompt
    | llm
    | StrOutputParser()
)
