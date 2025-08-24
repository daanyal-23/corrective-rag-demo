from tools import retrieval_grader

def grade_documents(state):
    """
    Determines whether the retrived documnets are relevant to the question
    """
    print("---CHECK DOCUMENT RELEVANCE TO THE QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE:DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE:DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}
