import os
import json
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# Configuration
VECTOR_DB_FILE = "./faiss_index.faiss"  # File for FAISS index
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama endpoint

# Initialize HuggingFace Embeddings and FAISS
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Small, efficient model

# Load or Create FAISS Vector Store
def create_faiss_vector_store(json_dir):
    documents = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename), 'r') as file:
                data = json.load(file)
                for key, value in data.items():
                    documents.append(value)

    # Create FAISS vector store
    faiss_index = FAISS.from_texts(documents, embedding=embedding_model)
    faiss_index.save_local(VECTOR_DB_FILE)
    return faiss_index

if not os.path.exists(VECTOR_DB_FILE):
    faiss_index = create_faiss_vector_store("./datasets/dataset")
else:
    faiss_index = FAISS.load_local(
        VECTOR_DB_FILE,
        embedding_model,
        allow_dangerous_deserialization=True
    )

# Query the Ollama server
def query_ollama(prompt):
    payload = {"prompt": prompt, "model": "llama3.1:8b"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(OLLAMA_API_URL, json=payload, headers=headers, stream=True)
    response.raise_for_status()
    
    full_response = ""
    for chunk in response.iter_lines():
        if chunk:
            chunk_data = json.loads(chunk.decode('utf-8'))
            full_response += chunk_data.get("response", "")
    return full_response.strip()

# RAG Node
def generate_answer(query):
    docs = faiss_index.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    return query_ollama(prompt)

# Recommendation Node
def recommend_or_warn(query):
    docs = faiss_index.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        f"Task: Provide recommendations, warnings, or alternatives, and explain the reasoning.\nAnswer:"
    )
    return query_ollama(prompt)

# Alternatives Node
def generate_alternatives(query):
    docs = faiss_index.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        f"Task: Suggest alternatives to the specified product and explain their advantages.\nAnswer:"
    )
    return query_ollama(prompt)

# Summarization Node
def generate_summary(query):
    docs = faiss_index.similarity_search(query, k=5)
    section_summaries = []
    for doc in docs:
        section_context = doc.page_content
        prompt = f"Context: {section_context}\n\nTask: Summarize the context into a concise summary.\nSummary:"
        section_summaries.append(query_ollama(prompt))
    return "\n".join(section_summaries).strip()

# Web Search Node (Placeholder)
def web_search(query):
    # Placeholder for web search implementation
    return f"Web search results for: {query}"

# Router Node
def handle_query(query):
    # Simple query classification
    if "alternatives" in query.lower() or "suggest" in query.lower():
        return generate_alternatives(query)
    elif "summarize" in query.lower():
        return generate_summary(query)
    elif any(keyword in query.lower() for keyword in ["can I", "suitable for", "recommend", "safe for"]):
        return recommend_or_warn(query)
    else:
        return generate_answer(query)

# Streamlit Interface
st.title("Pharma Knowledge Assistant")
query = st.text_input("Ask a question about pharmaceutical products:")
if query:
    st.write("Processing...")
    answer = handle_query(query)
    st.write("Answer:", answer)