from langgraph.graph import START, END, StateGraph
from src.state.graph_state import GraphState
from src.nodes import retrive, grade_documents, generate, transform_query, web_search

def decide_to_generate(state):
    """Determines whether to generate an answer or re-generate a question"""
    state["logs"].append("---ASSESS GRADED DOCUMENTS---")

    web_search_flag = state["web_search"]

    if web_search_flag == "Yes":
        state["logs"].append(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO THE QUESTION, TRANSFORM THE QUERY---"
        )
        return "transform_query"
    else:
        state["logs"].append("---DECISION: GENERATE---")
        return "generate"


def build_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrive)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)

    # Build the graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
