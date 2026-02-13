from src.tools.rag_resources import get_retriever
from UI.streamlitUI.execution_trace import ExecutionTrace


def retrive(state):
    trace = ExecutionTrace()

    question = state["question"]

    retriever = get_retriever()

    if retriever is None:
        state["documents"] = []
        trace.add_advanced_log("Retriever returned None.")
        return state

    docs = retriever.invoke(question)

    trace.add_step(
        "🔍 Retrieve Context",
        f"Retrieved {len(docs)} documents from available sources."
    )

    # 🔎 DEBUG: Preview first retrieved document safely
    if docs and hasattr(docs[0], "page_content"):
        preview = docs[0].page_content[:200].replace("\n", " ")
        trace.add_advanced_log(
            f"First retrieved doc preview: {preview}"
        )
    else:
        trace.add_advanced_log("No valid documents retrieved.")

    state["documents"] = docs
    return state
