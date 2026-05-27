from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


def build_chain():
    # Prompt setup
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a beginner-friendly programming instructor. Explain concepts in very simple language using relatable analogies."),
        ("human", "Explain {topic} using an analogy from {analogy_domain}. Keep it short and easy.")
    ])

    # LLM setup
    llm = ChatOllama(
        model="qwen:1.8b",
        base_url="http://localhost:11434",
        temperature=1,
        num_predict=100
    )

    # Output parser
    parser = StrOutputParser()

    # LCEL chain
    chain = prompt | llm | parser

    return chain
