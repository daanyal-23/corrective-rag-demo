from src.tools.rag_resources import get_rag_chain
from UI.streamlitUI.execution_trace import ExecutionTrace

MAX_CONTEXT_CHARS = 3000


def generate(state):
    trace = ExecutionTrace()

    question = state.get("question", "")
    docs = state.get("documents", [])

    context = "\n\n".join([doc.page_content for doc in docs])

    if len(context) > MAX_CONTEXT_CHARS:
        trace.add_advanced_log(
            f"Context too long ({len(context)} chars). Truncated."
        )

    try:
        chain = get_rag_chain()

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
