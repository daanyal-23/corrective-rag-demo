from langchain_core.prompts import ChatPromptTemplate
from src.tools.rag_resources import llm
from UI.streamlitUI.execution_trace import ExecutionTrace

MAX_CONTEXT_CHARS = 3000  # adjust based on Groq model context window


def generate(state):
    """
    Generate final answer using LLM.
    Args:
        state (dict): The current graph state containing documents and question.
    Returns:
        dict: Updated state with 'generation' key.
    """

    trace = ExecutionTrace()

    question = state.get("question", "")
    docs = state.get("documents", [])

    # Combine document contents into context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Truncate context if too long
    if len(context) > MAX_CONTEXT_CHARS:
        trace.add_advanced_log(
            f"Context too long ({len(context)} chars). Truncated to {MAX_CONTEXT_CHARS} chars."
        )
        context = context[:MAX_CONTEXT_CHARS] + "... [truncated]"

    # ⚠️ Do NOT truncate the question — log only
    if len(question) > 500:
        trace.add_advanced_log(
            f"Question length is high ({len(question)} chars). Passed to model as-is."
        )

    # Prepare prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the given context to answer the question."
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}"
            ),
        ]
    )

    try:
        chain = prompt | llm
        response = chain.invoke(
            {"question": question, "context": context}
        )

        generation = (
            response.content if hasattr(response, "content") else str(response)
        )

        trace.add_step(
            "✍️ Generate Answer",
            "Final response generated successfully."
        )

    except Exception as e:
        trace.add_advanced_log(f"Generation failed: {str(e)}")
        generation = "An error occurred while generating the response."

    state["generation"] = generation
    return state
