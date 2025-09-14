import json
from langchain_core.documents import Document
from src.tools.rag_resources import web_search_tool

def web_search(state):
    state["logs"].append("---WEB SEARCH---")
    query = state["question"]

    try:
        raw_results = web_search_tool.invoke({"query": query})
        state["logs"].append(f"---WEB SEARCH RESULTS STRUCTURE: {type(raw_results)}---")

        docs = []

        # Case 1: Tavily returned plain string
        if isinstance(raw_results, str):
            try:
                parsed = json.loads(raw_results)
                if isinstance(parsed, list):
                    for item in parsed[:3]:  # take top 3 only
                        content = item.get("content") or str(item)
                        docs.append(Document(page_content=content[:1000]))  # truncate each
                else:
                    docs.append(Document(page_content=str(parsed)[:1000]))
            except json.JSONDecodeError:
                docs.append(Document(page_content=raw_results[:1000]))

        # Case 2: Tavily returned list of dicts
        elif isinstance(raw_results, list):
            for item in raw_results[:3]:  # limit to top 3
                if isinstance(item, dict):
                    content = item.get("content") or str(item)
                    docs.append(Document(page_content=content[:1000]))
                else:
                    docs.append(Document(page_content=str(item)[:1000]))

        # Case 3: Tavily returned dict or other object
        elif isinstance(raw_results, dict):
            # Try to extract common keys
            snippet_keys = ["content", "text", "snippet", "summary"]
            snippets = []
            for key in snippet_keys:
                if key in raw_results:
                    snippets.append(str(raw_results[key]))
            if snippets:
                docs.append(Document(page_content="\n".join(snippets)[:1000]))
            else:
                docs.append(Document(page_content=str(raw_results)[:1000]))

        else:
            docs.append(Document(page_content=str(raw_results)[:1000]))

        state["documents"] = docs

    except Exception as e:
        state["logs"].append(f"Web search error: {e}")
        state["documents"] = [Document(page_content=f"Web search failed: {e}")]

    return state
