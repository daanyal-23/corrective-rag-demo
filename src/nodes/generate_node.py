from src.tools.rag_resources import get_rag_chain
from UI.streamlitUI.execution_trace import ExecutionTrace


def generate(state):
    """
    Generate final answer using RAG chain.
    """

    trace = ExecutionTrace()

    question = state.get("question", "")
    docs = state.get("documents", [])

    try:
        chain = get_rag_chain()

        # Pass documents directly — let the chain build context
        response = chain.invoke(
            {
                "question": question,
                "documents": docs,
            }
        )

        generation = str(response)

        trace.add_step(
            "✍️ Generate Answer",
            "Final response generated successfully."
        )

    except Exception as e:
        trace.add_advanced_log(f"Generation failed: {str(e)}")
        generation = "An error occurred while generating the response."

    state["generation"] = generation
    return state
