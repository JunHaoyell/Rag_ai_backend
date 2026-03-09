from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import ask_rag
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境可以用 *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask(q: Question):

    answer = ask_rag(q.question)

    return {
        "question": q.question,
        "answer": answer
    }