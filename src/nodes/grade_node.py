from src.tools.rag_resources import get_retrieval_grader
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

    # ✅ Lazy initialization (CI-safe)
    retrieval_grader = get_retrieval_grader()

    for d in documents:
        try:
            result = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )

            grade = result.get("binary_score", "no")

        except Exception:
            grade = "no"

        if grade == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"

    trace.add_step(
        "📄 Evaluate Retrieved Documents",
        f"{len(filtered_docs)} / {len(documents)} documents marked as relevant."
    )

    state["documents"] = filtered_docs
    state["web_search"] = web_search
    return state
