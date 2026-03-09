from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import ask_rag
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Question):
    answer = ask_rag(q.question)
    return {"question": q.question, "answer": answer}

@app.get("/")
def root():
    return {"message": "RAG AI Backend running ✅"}
