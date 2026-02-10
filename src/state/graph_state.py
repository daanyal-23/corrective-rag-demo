from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of the Corrective RAG graph.

    Attributes:
        question: User question
        generation: Final LLM-generated answer
        web_search: Flag indicating whether web search is required ("Yes" / "No")
        documents: List of retrieved documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]
