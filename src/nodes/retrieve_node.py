from src.tools import retriever

def retrive(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updated state with retrieved documents
    """
    # Log instead of print
    state["logs"].append("---RETRIEVE---")

    question = state["question"]
    documents = retriever.invoke(question)

    state["documents"] = documents
    return state
