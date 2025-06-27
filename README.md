# 📄 RAG-Based Document Question Answering

This project implements a **Retrieval-Augmented Generation (RAG)** system using [HuggingFace’s Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B) and **LangChain** to enable intelligent question answering over PDF documents.

---

## 🚀 Features

- 🧠 **LLM Integration**: Uses HuggingFace's Mixtral-8x7B via LangChain.
- 📄 **PDF Ingestion**: Supports multi-page document parsing.
- ✂️ **Text Chunking**: Splits documents into manageable, semantically coherent chunks.
- 🔍 **Embedding Generation**: Uses `sentence-transformers/all-mpnet-base-v2` for dense vector embeddings.
- 🧠 **Vector Store**: Stores embeddings in **ChromaDB** for efficient semantic search.
- ❓ **Question Answering**: Retrieves relevant chunks and generates responses using the LLM.

---

## 🛠 Tech Stack

- [LangChain](https://www.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/)
- [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B)
- [Sentence-Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [PyMuPDF / pdfplumber / fitz] (for PDF parsing)
- Python 3.8+

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Yuraj-Isurinda/RAG_QnA_Application.git
cd RAG_QnA_Application
```

## Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

## Install Dependencies
```bash
pip install -r requirements.txt
```
Add Your PDF
Place your PDFs in the docs/ folder (e.g., docs/TechBloom_Matha_1.0.pdf).


# usage 
```bash
# In your Jupyter notebook or main.py script
from rag_pipeline import process_pdf, query_pdf

# Step 1: Process PDF and build vector store
process_pdf("docs/TechBloom_Matha_1.0.pdf")

# Step 2: Ask a question
response = query_pdf("What is TechBloom Matha?")
print(response)
```

# Project Structure
```bash
rag-doc-qa/
│
├── docs/                                 # Vector store (ChromaDB)
├── Code.ipynb          # Main RAG pipeline code
├── requirements.txt         # Python dependencies
└── README.md
```

