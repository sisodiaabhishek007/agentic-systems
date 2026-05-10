import os
import re
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from pypdf import PdfReader


# =========================
# CONFIGURATION
# =========================

POLICY_FOLDER = "policy_documents"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "campus_policies"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 20

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"


# =========================
# OPENAI CLIENT
# =========================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY environment variable not found."
    )


# =========================
# POLICY TYPE INFERENCE
# =========================

def infer_policy_type(filename):
    """
    Infer policy type from filename.
    """

    name = filename.lower()

    if "hostel" in name:
        return "hostel"

    elif "refund" in name:
        return "refund"

    elif "library" in name:
        return "library"

    else:
        return "general"


# =========================
# CLEAN TEXT
# =========================

def clean_text(text):
    """
    Remove extra spaces and newlines.
    """

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# LOAD PDF DOCUMENTS
# =========================

def load_pdf_documents(folder_path):
    """
    Load all PDFs from policy_documents folder.
    """

    documents = []

    pdf_files = Path(folder_path).glob("*.pdf")

    for pdf_file in pdf_files:

        reader = PdfReader(str(pdf_file))

        print(f"Loaded {len(reader.pages)} pages from: {pdf_file.name}")

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text:
                cleaned = clean_text(text)

                documents.append({
                    "text": cleaned,
                    "source": pdf_file.name,
                    "page": page_number,
                    "policy_type": infer_policy_type(pdf_file.name)
                })

    return documents


# =========================
# TEXT CHUNKING
# =========================

def split_into_chunks(text,
                      chunk_size=CHUNK_SIZE,
                      overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        start += (chunk_size - overlap)

    return chunks


# =========================
# GENERATE EMBEDDINGS
# =========================

def generate_embedding(text):
    """
    Generate embedding using OpenAI embedding model.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


# =========================
# BUILD CHROMADB
# =========================

def build_vector_database(documents):
    """
    Build persistent ChromaDB collection.
    """

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    total_chunks = 0

    for doc in documents:

        chunks = split_into_chunks(doc["text"])

        for chunk in chunks:

            embedding = generate_embedding(chunk)

            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": doc["source"],
                    "page": doc["page"],
                    "policy_type": doc["policy_type"]
                }]
            )

            total_chunks += 1

    print(f"Total chunks created: {total_chunks}")
    print(
        f"Successfully stored {total_chunks} chunks in vector database."
    )

    return collection


# =========================
# RETRIEVAL
# =========================

def retrieve_relevant_chunks(collection, query, top_k=3):
    """
    Retrieve top-k relevant chunks from ChromaDB.
    """

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_chunks = results["documents"][0]

    print(f"Retrieved {len(retrieved_chunks)} relevant chunks.")

    return retrieved_chunks


# =========================
# PROMPT BUILDER
# =========================

def build_prompt(context_chunks, user_query):
    """
    Build RAG prompt using retrieved chunks.
    """

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a campus policy assistant.

Answer ONLY from the retrieved policy context.

If the answer is not present in the context,
say:
"I don't have that information."

Keep the answer short, simple, and student-friendly.

Retrieved Policy Context:
{context}

Student Question:
{user_query}
"""

    return prompt


# =========================
# GENERATE ANSWER
# =========================

def generate_answer(prompt):
    """
    Generate final answer using LLM.
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You answer only from provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# =========================
# END-TO-END QA FUNCTION
# =========================

def answer_question(collection, question):
    """
    Complete RAG pipeline.
    """

    print("\n" + "=" * 60)

    print(f"User Query: {question}")

    retrieved_chunks = retrieve_relevant_chunks(
        collection,
        question
    )

    prompt = build_prompt(
        retrieved_chunks,
        question
    )

    answer = generate_answer(prompt)

    print(f"Answer: {answer}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("Building knowledge base...\n")

    documents = load_pdf_documents(POLICY_FOLDER)

    collection = build_vector_database(documents)

    print(f"\nVector DB ready. Collection: {COLLECTION_NAME}")

    test_queries = [
        "Can I get a refund after dropping a course?",
        "What is the deadline for returning a library book?",
        "Are hostel visitors allowed on weekends?"
    ]

    for query in test_queries:
        answer_question(collection, query)
