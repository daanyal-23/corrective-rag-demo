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
        result = graph.invoke({"question": question, "logs": []})  # start with empty logs

        st.subheader("Execution Steps:")
        for log in result.get("logs", []):
            st.text(log)

        st.subheader("Generation:")
        st.write(result.get("generation", ""))
    except Exception as e:
        st.error(f"Error: {str(e)}")
