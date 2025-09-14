from src.tools import question_rewriter

def transform_query(state):
    """Transform the query to produce a better result"""
    state["logs"].append("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}
