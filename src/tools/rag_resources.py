import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://github.com/phaneendra2429/Agent-RAG/blob/main/testing.ipynb",
]


@st.cache_resource
def load_web_docs(urls):
    docs = []
    for url in urls:
        try:
            docs.extend(WebBaseLoader(url).load())
        except Exception as e:
            print(f"⚠️ Failed to load {url}: {e}")
    return docs

@st.cache_resource
def build_retriever():
    docs = load_web_docs(URLS)

    if not docs:
        print("⚠️ No documents loaded.")
        return None

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50,
    )

    doc_splits = text_splitter.split_documents(docs)

    doc_splits = [
        d for d in doc_splits
        if d.page_content and d.page_content.strip()
    ]

    if not doc_splits:
        print("⚠️ All document chunks empty.")
        return None

    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=get_embeddings(),
    )

    return vectorstore.as_retriever()


def get_retriever():
    return build_retriever()


def get_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Required for runtime execution."
        )

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
    )


system_prompt = """You are a grader assessing relevance.
Return JSON:
{"binary_score": "yes"} or {"binary_score": "no"}.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Document:\n{document}\n\nQuestion:\n{question}"),
    ]
)


def get_retrieval_grader():
    llm = get_groq_llm()
    return grade_prompt | llm | JsonOutputParser()


rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using provided context. "
            "If unknown, say you do not know. "
            "Keep answer concise (max 3 sentences).",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}",
        ),
    ]
)


def get_rag_chain():
    llm = get_groq_llm()

    return (
        {
            "context": lambda x: "\n\n".join(
                d.page_content for d in x.get("documents", [])
            ),
            "question": lambda x: x["question"],
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the question to improve web search. Preserve intent.",
        ),
        (
            "human",
            "Original question:\n{question}\n\nImproved:",
        ),
    ]
)


def get_question_rewriter():
    llm = get_groq_llm()
    return rewrite_prompt | llm | StrOutputParser()


def get_web_search_tool():
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY not set. Required for runtime execution."
        )

    return TavilySearch(
        k=3,
        api_key=api_key,
    )


def safe_web_search(query: str):
    try:
        tool = get_web_search_tool()
        result = tool.invoke(query)

        if isinstance(result, str):
            try:
                return json.loads(result)
            except Exception:
                return []

        return result

    except Exception as e:
        print("⚠️ Tavily search failed:", e)
        return []


embed = get_embeddings()
