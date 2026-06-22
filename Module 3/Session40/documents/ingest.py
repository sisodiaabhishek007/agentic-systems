import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hostel_policy_docs"


def main():
    # Delete old DB before re-ingest
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    loader = DirectoryLoader(
        "documents",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"Ingested {len(chunks)} chunks into Chroma.")


if __name__ == "__main__":
    main()
