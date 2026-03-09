import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
VECTOR_PATH = os.path.join(BASE_DIR, "vector_store")

documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_PATH, file), encoding="utf-8")
        documents.extend(loader.load())

# 文本切块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Embedding
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 构建 FAISS
db = FAISS.from_documents(chunks, embedding)

os.makedirs(VECTOR_PATH, exist_ok=True)
db.save_local(VECTOR_PATH)

print("✅ Vector database created at:", VECTOR_PATH)
