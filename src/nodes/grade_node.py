from src.tools import retrieval_grader
from UI.streamlitUI.execution_trace import ExecutionTrace


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """

    trace = ExecutionTrace()

    question = state["question"]
    documents = state.get("documents", [])

    filtered_docs = []
    web_search = "No"

    for d in documents:
        result = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )

        try:
            grade = result["binary_score"]
        except (KeyError, TypeError):
            grade = "no"

        if grade == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"

    # ✅ Human-readable execution trace (single summary step)
    trace.add_step(
        "📄 Evaluate Retrieved Documents",
        f"{len(filtered_docs)} / {len(documents)} documents marked as relevant."
    )

    state["documents"] = filtered_docs
    state["web_search"] = web_search
    return state
