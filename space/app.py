from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_huggingface import HuggingFaceEndpoint

app = FastAPI()

class Query(BaseModel):
    prompt: str
    temperature: float = 0.1
    max_tokens: int = 512

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

@app.post("/generate")
def generate(query: Query):
    return {"response": llm(query.prompt)}