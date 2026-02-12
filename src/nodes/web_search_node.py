from langchain_core.documents import Document
from src.tools.rag_resources import safe_web_search
from UI.streamlitUI.execution_trace import ExecutionTrace


def web_search(state):
    trace = ExecutionTrace()

    query = state["question"]

    trace.add_step(
        "🌐 Web Search",
        "Searching the web for additional relevant context."
    )

    results = safe_web_search(query)

    docs = []

    for item in results:
        if isinstance(item, dict):
            content = item.get("content") or item.get("snippet") or str(item)
        else:
            content = str(item)

        docs.append(Document(page_content=content))

    trace.add_step(
        "🌐 Web Search",
        f"Retrieved {len(docs)} results from web search."
    )

    state["documents"] = docs
    return state
