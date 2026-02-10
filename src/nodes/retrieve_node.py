from src.tools import retriever
from UI.streamlitUI.execution_trace import ExecutionTrace


def retrive(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updated state with retrieved documents
    """

    trace = ExecutionTrace()
    question = state["question"]

    # 🔴 Handle empty / unavailable retriever safely
    if retriever is None:
        trace.add_step(
            "🔍 Retrieve Context",
            "No local documents available — skipping retrieval."
        )
        state["documents"] = []
        return state

    documents = retriever.invoke(question)

    trace.add_step(
        "🔍 Retrieve Context",
        f"Retrieved {len(documents)} documents from available sources."
    )

    state["documents"] = documents
    return state
