# src/tools/rag_resources.py

import os
import json
import streamlit as st
from dotenv import load_dotenv

# -------------------------------
# Environment
# -------------------------------
# Streamlit Cloud loads secrets automatically.
# Local users can still use a .env file.
load_dotenv()

# -------------------------------
# Embeddings & Vector Store
# -------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# -------------------------------
# Web Document Loading (SAFE)
# -------------------------------
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

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50,
    )

    doc_splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=get_embeddings(),
    )

    return vectorstore.as_retriever()

# 🔑 This is what other modules import
retriever = build_retriever()

# -------------------------------
# Retrieval Grader
# -------------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

grader_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)

system_prompt = """You are a grader assessing the relevance of retrieved documents to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no'.
Respond ONLY with a JSON object:
{"binary_score": "yes"} or {"binary_score": "no"}.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document:\n\n{document}\n\nUser Question: {question}"),
    ]
)

retrieval_grader = grade_prompt | grader_llm | JsonOutputParser()

# -------------------------------
# RAG Generation Chain
# -------------------------------
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

rag_prompt = hub.pull("rlm/rag-prompt")

rag_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = rag_prompt | rag_llm | StrOutputParser()

# -------------------------------
# Question Rewriter
# -------------------------------
rewrite_system = """You are a question re-writer that converts an input question
into a better version optimized for web search.
Preserve the original intent.
"""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewrite_system),
        ("human", "Original question:\n\n{question}\n\nImproved question:"),
    ]
)

rewrite_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    model_kwargs={"tool_choice": "none"},
)

question_rewriter = rewrite_prompt | rewrite_llm | StrOutputParser()

# -------------------------------
# Web Search (Safe Wrapper)
# -------------------------------
from langchain_tavily import TavilySearch

web_search_tool = TavilySearch(
    k=3,
    api_key=os.getenv("TAVILY_API_KEY"),
)

def safe_web_search(query: str):
    try:
        result = web_search_tool.invoke(query)
        if isinstance(result, str):
            try:
                return json.loads(result)
            except Exception:
                return []
        return result
    except Exception as e:
        print("⚠️ Tavily search failed:", e)
        return []
