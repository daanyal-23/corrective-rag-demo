import streamlit as st
from dotenv import load_dotenv

from src.langgraphCorrectiveAI import build_graph
from UI.streamlitUI.execution_trace import ExecutionTrace
from UI.streamlitUI.display_result import show_generation

load_dotenv()

st.set_page_config(
    page_title="Corrective RAG Demo",
    page_icon="🛠️",
    layout="centered",
)

st.title("🛠️ Corrective RAG (LangGraph + Groq)")

# Input stays FIXED at the top
question = st.text_input("Enter your question:")

# Dedicated containers (layout anchors)
execution_container = st.container()
answer_container = st.container()

if question:
    with execution_container:
        st.markdown("### 🧭 Agent Execution (Live)")
        trace = ExecutionTrace()
        trace.clear()

    try:
        graph = build_graph()

        # Graph execution (live updates happen here)
        result = graph.invoke({"question": question})

        with answer_container:
            st.divider()
            show_generation(result.get("generation", ""))

    except Exception as e:
        st.error(f"Error: {str(e)}")
