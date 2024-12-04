import os
import json
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# Configuration
VECTOR_DB_FILE = "./faiss_index.faiss"  # File for FAISS index

# Initialize Sentence Transformer Model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use a small, efficient model

# Load JSON dataset and create FAISS vector store
def create_faiss_vector_store(json_dir):
    documents = []
    metadata_list = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename), 'r') as file:
                data = json.load(file)
                for key, value in data.items():
                    documents.append(value)
                    metadata_list.append({"file": filename, "section": key})

    # Generate FAISS index using HuggingFace embeddings
    faiss_index = FAISS.from_texts(documents, embedding=embedding_model)
    faiss_index.save_local(VECTOR_DB_FILE)
    return faiss_index

# Load or Create FAISS Vector Store
if not os.path.exists(VECTOR_DB_FILE):
    faiss_index = create_faiss_vector_store("./datasets/dataset")
else: 
    faiss_index = FAISS.load_local(
        VECTOR_DB_FILE, 
        embedding_model, 
        allow_dangerous_deserialization=True  # Ensure the index file is trusted
    )

# Streamlit Interface
st.title("Pharma Knowledge Assistant")
query = st.text_input("Ask a question about pharmaceutical products:")
if query:
    # Retrieve relevant documents
    docs = faiss_index.similarity_search(query, k=3)
    st.write("Relevant Results:")
    for doc in docs:
        st.write(doc)