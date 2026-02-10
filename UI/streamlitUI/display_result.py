import streamlit as st
from UI.streamlitUI.execution_trace import ExecutionTrace


def show_generation(generation):
    """
    Display the final generated answer.
    """
    st.subheader("📝 Generated Answer")
    st.write(generation)


def display_execution_trace():
    """
    Display the execution trace of the agentic workflow.
    """
    trace = ExecutionTrace()
    trace.render()
