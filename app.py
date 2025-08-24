import os
from dotenv import load_dotenv
import streamlit as st

# Backend graph
from src.langgraphCorrectiveAI import build_graph

load_dotenv()

st.set_page_config(page_title="Corrective RAG Demo", page_icon="🛠️", layout="centered")
st.title("🛠️ Corrective RAG (LangGraph + Groq)")

question = st.text_input("Enter your question:")

if question:
    try:
        graph = build_graph()
        # initial state: only question; the graph fills the rest exactly like your code
        result = graph.invoke({"question": question})
        st.subheader("Generation:")
        st.write(result.get("generation", ""))
    except Exception as e:
        st.error(f"Error: {str(e)}")
