from src.tools.rag_resources import get_question_rewriter
from UI.streamlitUI.execution_trace import ExecutionTrace

MAX_QUERY_CHARS = 350  # keep under API limit


def transform_query(state):
    """
    Transform the query to produce a better result when context is insufficient.
    """

    trace = ExecutionTrace()

    question = state.get("question", "")
    documents = state.get("documents", [])

    # 🧠 Decision: Do we need to transform the query?
    if documents:
        trace.add_step(
            "🧠 Decide Next Action",
            "Context sufficient → skipping query transformation."
        )
        return state

    trace.add_step(
        "🧠 Decide Next Action",
        "Context insufficient → rewriting query for better retrieval."
    )

    try:
        # ✅ LAZY INITIALIZATION (CI-safe)
        question_rewriter = get_question_rewriter()

        better_question = question_rewriter.invoke(
            {"question": question}
        )
        better_question = str(better_question)

        # Length safety
        if len(better_question) > MAX_QUERY_CHARS:
            trace.add_advanced_log(
                f"Rewritten query too long ({len(better_question)} chars). "
                f"Truncated to {MAX_QUERY_CHARS} chars."
            )
            better_question = (
                better_question[:MAX_QUERY_CHARS] + "... [truncated]"
            )

        state["question"] = better_question
        state["documents"] = documents

        trace.add_step(
            "✏️ Rewrite Query",
            "Query successfully rewritten to improve retrieval."
        )

    except Exception as e:
        trace.add_advanced_log(
            f"Query transformation failed: {str(e)}"
        )

        # Fallback to original
        state["question"] = question
        state["documents"] = documents

        trace.add_step(
            "✏️ Rewrite Query",
            "Query rewrite failed — falling back to original question."
        )

    return state
