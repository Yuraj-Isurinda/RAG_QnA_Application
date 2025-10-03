import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from my_rag import chain

app = FastAPI()

origins = [
    "http://localhost:5173/ask",
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
    except Exception as e:
        raise
    return AskRes(answer=answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)