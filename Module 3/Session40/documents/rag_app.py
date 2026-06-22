from operator import itemgetter

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hostel_policy_docs"


def format_docs(docs):
    formatted = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        formatted.append(
            f"Source: {source}\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def build_chain():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a hostel policy assistant.

Rules:
1. Use only the retrieved context.
2. If the answer is not present in the context, respond exactly:
   I don't know based on the provided documents.
3. Mention the source file name when possible.
4. Do not make up facts.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = (
        {
            "context": itemgetter("question")
            | retriever
            | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
        | StrOutputParser()
    )

    return chain


def main():
    chain = build_chain()

    q1 = "What are the quiet hours on weekdays?"
    q2 = "What is the scholarship amount for hostel residents?"

    print("=" * 60)
    print("Q1:", q1)
    print("Answer:")
    print(chain.invoke({"question": q1}))

    print("\n" + "=" * 60)
    print("Q2:", q2)
    print("Answer:")
    print(chain.invoke({"question": q2}))


if __name__ == "__main__":
    main()
