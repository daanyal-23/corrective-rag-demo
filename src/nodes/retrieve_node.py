from src.tools import retriever

def retrive(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updated state with retrieved documents
    """

    # Ensure logs exist
    if "logs" not in state:
        state["logs"] = []

    state["logs"].append("---RETRIEVE---")

    question = state["question"]

    # 🔴 CRITICAL FIX: Handle empty / unavailable retriever
    if retriever is None:
        state["logs"].append("⚠️ Retriever unavailable, returning empty documents.")
        state["documents"] = []
        return state

    documents = retriever.invoke(question)
    state["documents"] = documents
    return state
