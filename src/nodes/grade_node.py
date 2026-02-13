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

    # 🔥 Lazy initialization (CI-safe)
    retrieval_grader = get_retrieval_grader()

    for idx, d in enumerate(documents):
        try:
            result = retrieval_grader.invoke(
                {
                    "question": question,
                    "document": d.page_content
                }
            )

            # 🧠 DEBUG: Log raw grader output
            trace.add_advanced_log(
                f"[Doc {idx}] Grader raw output: {result}"
            )

            grade = str(result.get("binary_score", "no")).strip().lower()

        except Exception as e:
            trace.add_advanced_log(
                f"[Doc {idx}] Grader error: {str(e)}"
            )
            grade = "no"

        if grade == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"

    # ✅ Summary trace (clean UI)
    trace.add_step(
        "📄 Evaluate Retrieved Documents",
        f"{len(filtered_docs)} / {len(documents)} documents marked as relevant."
    )

    # 🧠 Optional Debug Preview
    if filtered_docs:
        preview = filtered_docs[0].page_content[:120].replace("\n", " ")
        trace.add_advanced_log(
            f"First relevant doc preview: {preview}"
        )

    state["documents"] = filtered_docs
    state["web_search"] = web_search

    return state
