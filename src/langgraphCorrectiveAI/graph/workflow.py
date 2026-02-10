from langgraph.graph import START, END, StateGraph
from src.state.graph_state import GraphState
from src.nodes import (
    retrive,
    grade_documents,
    generate,
    transform_query,
    web_search,
)
from UI.streamlitUI.execution_trace import ExecutionTrace


def decide_to_generate(state):
    """
    Determines whether to generate an answer or transform the query.
    """

    trace = ExecutionTrace()

    web_search_flag = state.get("web_search", "No")

    trace.add_step(
        "🧠 Decide Next Action",
        "Assessing whether retrieved context is sufficient for generation."
    )

    if web_search_flag == "Yes":
        trace.add_step(
            "🧠 Decide Next Action",
            "Retrieved documents insufficient → transforming query."
        )
        return "transform_query"
    else:
        trace.add_step(
            "🧠 Decide Next Action",
            "Retrieved documents sufficient → proceeding to generation."
        )
        return "generate"


def build_graph():
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("retrieve", retrive)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)

    # Build graph edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
