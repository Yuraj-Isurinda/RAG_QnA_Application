import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from my_rag import chain, rag_store

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskReq(BaseModel):
    question: str
    history: list = []

class AskRes(BaseModel):
    answer: str

@app.post("/ask", response_model=AskRes)
def ask(req: AskReq):
    payload = {"question": req.question, "history": req.history}
    try:
        answer = chain.invoke(payload)
        return AskRes(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def list_documents():
    # Keeps compatibility with your frontend (expects simple array of names)
    return {"documents": rag_store.list_documents()}

@app.get("/documents/detail")
def list_documents_detail():
    # If you build a manage UI, this gives doc_id for deletion
    return {"documents": rag_store.list_documents_detailed()}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    save_dir = "./data/uploads"
    os.makedirs(save_dir, exist_ok=True)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
    save_path = os.path.join(save_dir, file.filename)

    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        ok, msg, doc_id = rag_store.add_pdf(save_path)
    except Exception as e:
        # This will catch unexpected exceptions inside add_pdf (like persist errors)
        raise HTTPException(status_code=500, detail=f"Indexing error: {e}")

    if not ok:
        raise HTTPException(status_code=500, detail=msg)

    return {"success": True, "message": msg, "filename": file.filename, "doc_id": doc_id}

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    ok, msg = rag_store.remove_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail=msg)
    return {"success": True, "message": msg, "doc_id": doc_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)