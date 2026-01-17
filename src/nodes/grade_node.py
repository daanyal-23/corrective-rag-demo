import json
from src.tools import retrieval_grader

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    state["logs"].append("---CHECK DOCUMENT RELEVANCE TO THE QUESTION---")

    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        try:
            grade = score["binary_score"]
        except (KeyError, TypeError):
            # Fallback to 'no' if parsing fails
            grade = "no"

        if grade == "yes":
            state["logs"].append("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            state["logs"].append("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue

    state["documents"] = filtered_docs
    state["web_search"] = web_search
    return state
