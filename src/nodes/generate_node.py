from langchain_core.prompts import ChatPromptTemplate
from src.tools.rag_resources import llm

MAX_CONTEXT_CHARS = 3000  # adjust based on Groq model context window

def generate(state):
    """
    Generate final answer using LLM.
    Args:
        state (dict): The current graph state containing documents and question.
    Returns:
        dict: Updated state with 'generation' key.
    """
    state["logs"].append("---GENERATE---")
    question = state.get("question", "")
    docs = state.get("documents", [])

    # Combine document contents into context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Truncate context if too long
    if len(context) > MAX_CONTEXT_CHARS:
        state["logs"].append(f"Context too long ({len(context)} chars). Truncating...")
        context = context[:MAX_CONTEXT_CHARS] + "... [truncated]"

    # ✅ Do NOT truncate the question — keep original user input
    # If you want safety, just log very long ones without modifying:
    if len(question) > 500:
        state["logs"].append(
            f"Warning: Question is long ({len(question)} chars), passing as-is."
        )

    # Prepare prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use the given context to answer the question."),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ]
    )

    try:
        chain = prompt | llm
        response = chain.invoke({"question": question, "context": context})
        generation = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        generation = f"Generation failed: {e}"
        state["logs"].append(generation)

    state["generation"] = generation
    return state
