from src.tools import question_rewriter

MAX_QUERY_CHARS = 350  # keep under the 400-char API limit

def transform_query(state):
    """Transform the query to produce a better result"""
    state["logs"].append("---TRANSFORM QUERY---")
    question = state.get("question", "")
    documents = state.get("documents", [])

    try:
        better_question = question_rewriter.invoke({"question": question})
        better_question = str(better_question)

        # Enforce query length safety
        if len(better_question) > MAX_QUERY_CHARS:
            state["logs"].append(
                f"Transformed query too long ({len(better_question)} chars). Truncating..."
            )
            better_question = better_question[:MAX_QUERY_CHARS] + "... [truncated]"

        state["question"] = better_question
        state["documents"] = documents

    except Exception as e:
        state["logs"].append(f"Query transformation failed: {e}")
        # Fallback: keep original question
        state["question"] = question
        state["documents"] = documents

    return state
