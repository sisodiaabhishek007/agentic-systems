
import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ==============================
# OpenAI Client
# ==============================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==============================
# HR Policy Documents
# ==============================

HR_POLICY_DOCUMENTS = [
    {
        "id": "policy_001",
        "text": """
        Employees are entitled to 24 days of annual leave every calendar year.
        Unused leave can be carried forward up to a maximum of 10 days into the next year.
        Sick leave of up to 12 days annually is available and may require medical documentation for absences longer than 3 consecutive days.
        Leave requests must be submitted through the HR portal at least 5 working days in advance whenever possible.
        Emergency leave situations should be communicated directly to the reporting manager and HR team.
        """,
        "metadata": {
            "category": "Leave Policy",
            "source": "HR Handbook v2026"
        }
    },
    {
        "id": "policy_002",
        "text": """
        Employees who have completed at least 6 months with the company are eligible for the Work From Home program.
        Eligible employees may work remotely up to 3 days per week with manager approval.
        Work from home requests must be submitted at least one day in advance through the internal HR system.
        Employees are expected to remain available during official working hours and attend all scheduled virtual meetings.
        The company reserves the right to revoke remote work privileges in case of policy misuse or performance concerns.
        """,
        "metadata": {
            "category": "Work From Home Policy",
            "source": "Remote Work Guidelines 2026"
        }
    },
    {
        "id": "policy_003",
        "text": """
        The company conducts employee appraisals once every year during the month of March.
        Performance is evaluated using a five-point rating scale ranging from Outstanding to Needs Improvement.
        Salary increments and bonus eligibility are determined based on appraisal ratings, business performance, and manager feedback.
        Employees are encouraged to complete self-assessment forms before appraisal discussions.
        Final appraisal decisions are reviewed and approved by the HR department and senior leadership team.
        """,
        "metadata": {
            "category": "Appraisal Policy",
            "source": "Performance Management Policy 2026"
        }
    },
    {
        "id": "policy_004",
        "text": """
        Employees are expected to maintain professional and respectful behavior in the workplace at all times.
        Sharing confidential company or customer data with unauthorized individuals is strictly prohibited.
        Employees must avoid conflicts of interest that could impact business decisions or company reputation.
        Harassment, discrimination, or unethical conduct may result in disciplinary action including termination.
        All employees are required to comply with company security and data privacy guidelines.
        """,
        "metadata": {
            "category": "Code of Conduct",
            "source": "Employee Code of Ethics 2026"
        }
    }
]

# ==============================
# Embedding Function
# ==============================

def create_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [item.embedding for item in response.data]
    return embeddings

# ==============================
# Setup ChromaDB
# ==============================

def setup_vector_database():

    chroma_client = chromadb.PersistentClient(
        path="./chroma_hr_policy_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="hr_policy_collection",
        metadata={"hnsw:space": "cosine"}
    )

    return collection

# ==============================
# Index Documents
# ==============================

def index_hr_documents(collection):

    texts = [doc["text"] for doc in HR_POLICY_DOCUMENTS]
    embeddings = create_embeddings(texts)

    collection.upsert(
        ids=[doc["id"] for doc in HR_POLICY_DOCUMENTS],
        documents=texts,
        metadatas=[doc["metadata"] for doc in HR_POLICY_DOCUMENTS],
        embeddings=embeddings
    )

    print("\nDocuments indexed successfully.\n")

# ==============================
# Retrieve Content
# ==============================

def retrieve_hr_content(collection, query, top_k=3):

    query_embedding = create_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_chunks = []

    for i in range(len(results["ids"][0])):
        chunk = {
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        }

        retrieved_chunks.append(chunk)

    return retrieved_chunks

# ==============================
# Build Grounded Prompt
# ==============================

def build_grounded_prompt(query, chunks):

    context = ""

    for idx, chunk in enumerate(chunks, start=1):
        context += f"""
        Context Chunk {idx}:
        Category: {chunk['metadata']['category']}
        Source: {chunk['metadata']['source']}

        {chunk['text']}
        """

    prompt = f"""
    You are an HR Policy Assistant for InnoTech Solutions.

    Answer the employee question ONLY using the policy context provided below.

    If the answer is not available in the context, say:
    "The HR policy documents do not contain information about this."

    Do not guess or make up policies.

    ========================
    POLICY CONTEXT
    ========================

    {context}

    ========================
    EMPLOYEE QUESTION
    ========================

    {query}

    ========================
    FINAL ANSWER
    ========================
    """

    return prompt

# ==============================
# Generate Answer With RAG
# ==============================

def generate_answer(query, chunks):

    grounded_prompt = build_grounded_prompt(query, chunks)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful HR assistant."
            },
            {
                "role": "user",
                "content": grounded_prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

# ==============================
# Generate Answer Without RAG
# ==============================

def generate_answer_without_retrieval(query):

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful HR assistant."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

# ==============================
# Complete RAG Pipeline
# ==============================

def answer_with_rag(collection, query, top_k=3):

    print("\n" + "=" * 80)
    print(f"QUESTION: {query}")
    print("=" * 80)

    retrieved_chunks = retrieve_hr_content(collection, query, top_k)

    print("\nRetrieved Chunks:\n")

    for idx, chunk in enumerate(retrieved_chunks, start=1):

        print(f"Chunk {idx}")
        print(f"Category : {chunk['metadata']['category']}")
        print(f"Source   : {chunk['metadata']['source']}")
        print(f"Distance : {chunk['distance']:.4f}")
        print(f"Text     : {chunk['text']}")
        print("-" * 80)

    final_answer = generate_answer(query, retrieved_chunks)

    print("\nFinal Answer:\n")
    print(final_answer)

# ==============================
# Main Execution
# ==============================

if __name__ == "__main__":

    collection = setup_vector_database()

    index_hr_documents(collection)

    # ==========================
    # Test Queries
    # ==========================

    queries = [
        "How many days of annual leave am I entitled to per year?",
        "Do I need manager approval before working from home?",
        "When is the appraisal cycle conducted and how is increment decided?"
    ]

    for query in queries:
        answer_with_rag(collection, query)

    # ==========================
    # Side-by-Side Comparison
    # ==========================

    comparison_query = "Can employees work remotely without approval?"

    print("\n" + "#" * 80)
    print("ANSWER WITHOUT RAG")
    print("#" * 80)

    no_rag_answer = generate_answer_without_retrieval(comparison_query)
    print(no_rag_answer)

    print("\n" + "#" * 80)
    print("ANSWER WITH RAG")
    print("#" * 80)

    answer_with_rag(collection, comparison_query)
