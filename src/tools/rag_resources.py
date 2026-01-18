# src/tools/rag_resources.py

import os
import json
from dotenv import load_dotenv
# --- Embeddings, loaders, vectorstore, retriever ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # updated

load_dotenv(dotenv_path="C:/Users/Mr. Daanyal/Desktop/Agentic_Bootcamp/CorrectiveRAG/.env")


embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://github.com/phaneendra2429/Agent-RAG/blob/main/testing.ipynb",
]

# Load docs from URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# Vectorstore setup
vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embed)
retriever = vectorstore.as_retriever()

# --- Retrieval Grader setup ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")).bind_tools([])

system = """You are a grader assessing the relevance of retrieved documents to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Output your response as a JSON object in the format: {{"binary_score": "yes"}} or {{"binary_score": "no"}}.
Do not use function calls or tool calls. Respond only with the JSON object."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User Question: {question}"),
    ]
)

retrieval_grader = grade_prompt | llm | JsonOutputParser()

# --- RAG generate chain ---
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

prompt = hub.pull("rlm/rag-prompt")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()

# --- Rewriting the Question ---
system_rewrite = """You are a question re-writer that converts an input question
to a better version that is optimized for web search.
Look at the input and try to reason about the underlying semantic intent/meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        (
            "human",
            "Here is the initial question:\n\n {question} \n Formulate an improved question.",
        ),
    ]
)

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), model_kwargs={"tool_choice": "none"})
question_rewriter = re_write_prompt | llm | StrOutputParser()

# --- Web search tool (with safe wrapper) ---
from langchain_tavily import TavilySearch  # updated

web_search_tool = TavilySearch(k=3, api_key=os.getenv("TAVILY_API_KEY"))

def safe_web_search(query: str):
    """Run a safe Tavily search that won’t crash if API fails."""
    try:
        result = web_search_tool.invoke(query)
        # Defensive: Tavily should return dict/list, but fallback if it's a string
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                print("⚠️ Tavily returned raw string instead of JSON:", result)
                return []
        return result
    except Exception as e:
        print("⚠️ Web search failed:", e)
        return []
