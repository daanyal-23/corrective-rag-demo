from src.tools.rag_resources import get_retriever
from UI.streamlitUI.execution_trace import ExecutionTrace


def retrive(state):
    trace = ExecutionTrace()

    question = state["question"]

    retriever = get_retriever()

    if retriever is None:
        state["documents"] = []
        return state

    docs = retriever.invoke(question)

    trace.add_step(
        "🔍 Retrieve Context",
        f"Retrieved {len(docs)} documents from available sources."
    )

    state["documents"] = docs
    return state
