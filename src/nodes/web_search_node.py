import json
from langchain_core.documents import Document
from tools.rag_resources import web_search_tool

def web_search(state):
    print("---WEB SEARCH---")
    query = state["question"]

    try:
        raw_results = web_search_tool.invoke({"query": query})
        print("---WEB SEARCH RESULTS STRUCTURE---")
        print(type(raw_results))

        docs = []

        # Case 1: Tavily returned plain string
        if isinstance(raw_results, str):
            try:
                # try parsing as JSON
                parsed = json.loads(raw_results)
                if isinstance(parsed, list):
                    for item in parsed:
                        content = item.get("content") or str(item)
                        docs.append(Document(page_content=content))
                else:
                    docs.append(Document(page_content=str(parsed)))
            except json.JSONDecodeError:
                # just wrap raw string
                docs.append(Document(page_content=raw_results))

        # Case 2: Tavily returned list of dicts
        elif isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict):
                    content = item.get("content") or str(item)
                    docs.append(Document(page_content=content))
                else:
                    docs.append(Document(page_content=str(item)))

        # Case 3: Tavily returned something else (dict/object)
        else:
            docs.append(Document(page_content=str(raw_results)))

        state["documents"] = docs

    except Exception as e:
        print("Web search error:", e)
        state["documents"] = [Document(page_content=f"Web search failed: {e}")]

    return state
