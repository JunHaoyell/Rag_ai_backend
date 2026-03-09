from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "../vector_store",
    embedding,
    allow_dangerous_deserialization=True
)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def ask_rag(question):

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [f"文档{i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
你是一个企业知识库AI助手。

请严格根据提供的文档回答问题：
- 如果文档中有答案，请总结回答。
- 如果文档中没有相关信息，请回答："文档中没有相关信息"。
- 不要编造答案。

文档内容：
{context}

用户问题：
{question}
"""

    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant"
    )

    return response.choices[0].message.content