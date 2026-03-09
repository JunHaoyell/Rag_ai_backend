from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

data_path = "../data"

documents = []

for file in os.listdir(data_path):

    loader = TextLoader(
        os.path.join(data_path, file),
        encoding="utf-8"
    )

    docs = loader.load()
    documents.extend(docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embedding)

db.save_local("../vector_store")

print("Vector database created")