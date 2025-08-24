from tools import retriever

def retrive(state):
    """
    Retreive documnets
    Args:
        state(dict): The current graph state
    returns:
        state(dict): New key added to the state, documents, that contains retrived documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
