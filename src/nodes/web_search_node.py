import json
from langchain_core.documents import Document
from src.tools.rag_resources import web_search_tool
from UI.streamlitUI.execution_trace import ExecutionTrace

MAX_DOC_CHARS = 1500  # limit per document snippet
MAX_DOCS = 3  # number of docs to keep


def truncate_content(content: str) -> str:
    """Truncate long document content safely with a marker."""
    if len(content) > MAX_DOC_CHARS:
        return content[:MAX_DOC_CHARS] + "... [truncated]"
    return content


def web_search(state):
    """
    Perform web search when retrieved documents are insufficient.
    """

    trace = ExecutionTrace()
    query = state.get("question", "")

    trace.add_step(
        "🌐 Web Search",
        "Searching the web for additional relevant context."
    )

    try:
        raw_results = web_search_tool.invoke({"query": query})
        docs = []

        # Case 1: Tavily returned plain string
        if isinstance(raw_results, str):
            try:
                parsed = json.loads(raw_results)
                if isinstance(parsed, list):
                    for item in parsed[:MAX_DOCS]:
                        content = str(item.get("content") or item)
                        docs.append(
                            Document(page_content=truncate_content(content))
                        )
                else:
                    docs.append(
                        Document(page_content=truncate_content(str(parsed)))
                    )
            except json.JSONDecodeError:
                docs.append(
                    Document(page_content=truncate_content(raw_results))
                )

        # Case 2: Tavily returned list
        elif isinstance(raw_results, list):
            for item in raw_results[:MAX_DOCS]:
                if isinstance(item, dict):
                    content = str(item.get("content") or item)
                else:
                    content = str(item)
                docs.append(
                    Document(page_content=truncate_content(content))
                )

        # Case 3: Tavily returned dict/object
        elif isinstance(raw_results, dict):
            snippet_keys = ["content", "text", "snippet", "summary"]
            snippets = [
                str(raw_results[key])
                for key in snippet_keys
                if key in raw_results
            ]
            if snippets:
                docs.append(
                    Document(
                        page_content=truncate_content("\n".join(snippets))
                    )
                )
            else:
                docs.append(
                    Document(page_content=truncate_content(str(raw_results)))
                )

        # Fallback case
        else:
            docs.append(
                Document(page_content=truncate_content(str(raw_results)))
            )

        if not docs:
            trace.add_step(
                "🌐 Web Search",
                "No relevant web results found."
            )
        else:
            trace.add_step(
                "🌐 Web Search",
                f"Retrieved {len(docs)} results from web search."
            )

        state["documents"] = docs

    except Exception as e:
        trace.add_advanced_log(f"Web search error: {str(e)}")
        state["documents"] = []

    return state
