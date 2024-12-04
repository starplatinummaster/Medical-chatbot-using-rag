import os
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import requests

# Configuration
VECTOR_DB_FILE = "./faiss_index.faiss"  # File for FAISS index
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Correct Ollama endpoint

# Initialize HuggingFace Embeddings and FAISS
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

# Query the Ollama server (handling streaming)
def query_ollama(prompt):
    payload = {
        "prompt": prompt,
        "model": "llama3.1:8b"  # Ensure the correct model name
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, headers=headers, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Accumulate the response chunks
        full_response = ""
        for chunk in response.iter_lines():
            if chunk:  # Ensure there's data in the chunk
                try:
                    # Decode the chunk as JSON
                    chunk_data = json.loads(chunk.decode('utf-8'))
                    # Append the response part
                    full_response += chunk_data.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"Error decoding chunk: {e}")

        return full_response.strip()

    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama server: {e}"

# Function to generate alternatives
def generate_alternatives(query):
    # Retrieve relevant documents using RAG
    docs = faiss_index.similarity_search(query, k=5)
    context = "\n".join([doc.page_content for doc in docs])

    # Define an alternatives-specific prompt template
    prompt_template = (
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        f"Task: Suggest alternatives to the specified product. Explain why the alternatives are appropriate and provide their advantages, if applicable.\n"
        f"Alternatives:"
    )

    # Query the LLM for alternatives
    return query_ollama(prompt_template)

# Function to process query and generate a normal RAG-based response
def generate_answer(query):
    # Retrieve relevant documents
    docs = faiss_index.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Use prompt template for answer generation
    prompt_template = (
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # Generate answer using Ollama Llama
    return query_ollama(prompt_template)

# Function to decide query type and route appropriately
def handle_query(query):
    # Simple keyword-based query classification
    if "alternatives" in query.lower() or "suggest" in query.lower():
        return generate_alternatives(query)
    elif "summarize" in query.lower():
        return generate_summary(query)
    elif any(keyword in query.lower() for keyword in ["can I", "suitable for", "if I have", "recommend", "safe for"]):
        return recommend_or_warn(query)
    else:
        return generate_answer(query)

# Function to provide a summary of retrieved documents
def generate_summary(query):
    # Retrieve relevant documents
    docs = faiss_index.similarity_search(query, k=5)
    section_summaries = []

    # Summarize each document section
    for doc in docs:
        section_context = doc.page_content
        prompt_template = (
            f"Context: {section_context}\n\n"
            f"Task: Summarize the above context into a concise, user-friendly summary.\n"
            f"Summary:"
        )
        summary = query_ollama(prompt_template)
        section_summaries.append(summary)

    # Combine all section summaries
    combined_summary = "\n".join(section_summaries)
    return combined_summary.strip()

# Function to process query and provide recommendations or warnings
def recommend_or_warn(query):
    # Retrieve relevant documents using RAG
    docs = faiss_index.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Define a recommendation-specific prompt template
    prompt_template = (
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        f"Task: Determine if there are any warnings, risks, or contraindications related to the query. "
        f"If applicable, suggest alternative products or solutions and explain why.\n"
        f"Answer:"
    )

    # Query the LLM for recommendations
    return query_ollama(prompt_template)

# Streamlit Interface
st.title("Pharma Knowledge Assistant")
query = st.text_input("Ask a question about pharmaceutical products:")
if query:
    answer = handle_query(query)
    st.write("Answer:", answer)