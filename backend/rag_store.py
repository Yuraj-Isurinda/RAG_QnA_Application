# rag_store.py
import os, json, uuid, hashlib
from typing import List, Dict, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
UPLOADS_DIR      = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR       = os.path.join(DATA_DIR, "chroma_db")
DOC_INDEX_PATH   = os.path.join(DATA_DIR, "documents.json")
COLLECTION_NAME  = "my_collection"

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

def _load_index() -> Dict[str, Dict]:
    if not os.path.exists(DOC_INDEX_PATH):
        return {}
    with open(DOC_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_index(idx: Dict[str, Dict]):
    with open(DOC_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

class RAGStore:
    def __init__(self, google_api_key: str):
        self.emb = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",  # keep consistent!
            google_api_key=GOOGLE_API_KEY,
        )
        self.vs = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=self.emb,
        )
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        self.doc_index = _load_index()

    # --- Public API ---
    def list_documents(self) -> List[str]:
        return [d["display_name"] for d in self.doc_index.values()]

    def list_documents_detailed(self) -> List[Dict]:
        return list(self.doc_index.values())

    def retriever(self, k=4):
        return self.vs.as_retriever(search_kwargs={"k": k})

    def add_pdf(self, uploaded_path: str) -> Tuple[bool, str, str]:
        if not uploaded_path.lower().endswith(".pdf"):
            return False, "Only PDF is allowed", ""
        file_hash = _file_sha1(uploaded_path)
        doc_id = file_hash[:16]
        display_name = os.path.basename(uploaded_path)

        if doc_id in self.doc_index:
            return True, f"{display_name} already indexed", doc_id

        splits = self._load_and_split(uploaded_path)
        if not splits:
            return False, "Could not extract text (is it a scanned PDF?)", ""

        texts, metas, ids = [], [], []
        for i, d in enumerate(splits):
            page = d.metadata.get("page", i)
            texts.append(d.page_content)
            metas.append({"doc_id": doc_id, "source": display_name, "page": page})
            ids.append(f"{doc_id}:{page}:{uuid.uuid4()}")

        self.vs.add_texts(texts=texts, metadatas=metas, ids=ids)

        self.doc_index[doc_id] = {
            "doc_id": doc_id,
            "display_name": display_name,
            "path": uploaded_path,
            "num_chunks": len(texts),
        }
        _save_index(self.doc_index)
        return True, f"Ingested {display_name} ({len(texts)} chunks)", doc_id

    def remove_document(self, doc_id: str) -> Tuple[bool, str]:
        if doc_id not in self.doc_index:
            return False, "Document not found"

        self.vs.delete(where={"doc_id": doc_id})

        path = self.doc_index[doc_id]["path"]
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                pass  # file locked; ignore

        del self.doc_index[doc_id]
        _save_index(self.doc_index)
        return True, "Document removed"

    # --- Helpers ---
    def _load_and_split(self, pdf_path: str):
        docs = PyPDFLoader(pdf_path).load()
        if not docs:
            return []
        return self.splitter.split_documents(docs)

def build_store(google_api_key: str) -> RAGStore:
    return RAGStore(google_api_key)
