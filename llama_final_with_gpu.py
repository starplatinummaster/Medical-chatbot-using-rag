import os
import json
import requests
import faiss
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# Configuration
VECTOR_DB_FILE = "./faiss_index.faiss"  # File for FAISS index
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama endpoint

# Initialize HuggingFace Embeddings with GPU
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    embedding_function=SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
)

# Load or Create FAISS Vector Store
def create_faiss_vector_store(json_dir):
    documents = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename), 'r') as file:
                data = json.load(file)
                for key, value in data.items():
                    documents.append({"text": value, "metadata": {"file": filename}})
    
    # Initialize FAISS with GPU
    res = faiss.StandardGpuResources()
    cpu_index = FAISS.from_texts(
        [doc["text"] for doc in documents],
        embedding=embedding_model,
        metadatas=[doc["metadata"] for doc in documents]
    ).index
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    return FAISS(gpu_index, embedding_model)

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

# Node: RAG-based Answer with Sources
def generate_answer_with_sources(query):
    docs = faiss_index.similarity_search_with_score(query, k=3)
    sources = []
    context = ""

    for doc, score in docs:
        context += doc.page_content + "\n"
        sources.append(doc.metadata.get("file", "Unknown Source"))

    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    answer = query_ollama(prompt)
    source_list = "\nSources:\n" + "\n".join(set(sources))
    return answer + source_list

# Node: Recommendations or Warnings
def recommend_or_warn(query):
    docs = faiss_index.similarity_search_with_score(query, k=3)
    sources = []
    context = ""

    for doc, score in docs:
        context += doc.page_content + "\n"
        sources.append(doc.metadata.get("file", "Unknown Source"))

    prompt = (
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        f"Task: Provide recommendations, warnings, or alternatives, and explain the reasoning.\nAnswer:"
    )
    answer = query_ollama(prompt)
    source_list = "\nSources:\n" + "\n".join(set(sources))
    return answer + source_list

# Node: Alternatives Generator
def generate_alternatives(query):
    docs = faiss_index.similarity_search_with_score(query, k=5)
    sources = []
    context = ""

    for doc, score in docs:
        context += doc.page_content + "\n"
        sources.append(doc.metadata.get("file", "Unknown Source"))

    prompt = (
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        f"Task: Suggest alternatives to the specified product and explain their advantages.\nAnswer:"
    )
    answer = query_ollama(prompt)
    source_list = "\nSources:\n" + "\n".join(set(sources))
    return answer + source_list

# Node: Summarization
def generate_summary(query):
    docs = faiss_index.similarity_search(query, k=5)
    section_summaries = []
    sources = []

    for doc in docs:
        section_context = doc.page_content
        sources.append(doc.metadata.get("file", "Unknown Source"))
        prompt = f"Context: {section_context}\n\nTask: Summarize the context into a concise summary.\nSummary:"
        section_summaries.append(query_ollama(prompt))

    combined_summary = "\n".join(section_summaries).strip()
    source_list = "\nSources:\n" + "\n".join(set(sources))
    return combined_summary + source_list

# Router Node
def handle_query(query):
    if "alternatives" in query.lower() or "suggest" in query.lower():
        return generate_alternatives(query)
    elif "summarize" in query.lower():
        return generate_summary(query)
    elif any(keyword in query.lower() for keyword in ["can I", "suitable for", "recommend", "safe for"]):
        return recommend_or_warn(query)
    else:
        return generate_answer_with_sources(query)

# Streamlit Interface
st.title("Pharma Knowledge Assistant")
query = st.text_input("Ask a question about pharmaceutical products:")
if query:
    st.write("Processing...")
    answer = handle_query(query)
    st.write("Answer:", answer)