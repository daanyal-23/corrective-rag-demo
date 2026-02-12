from src.langgraphCorrectiveAI.graph.workflow import build_graph


def test_graph_generates_without_web(monkeypatch):
    """
    Full graph flow test:
    - Retrieval returns a document
    - Grader marks it relevant
    - Decision routes directly to generate
    - Generate node is mocked to avoid real LLM call
    """

    # -----------------------
    # Mock Retriever
    # -----------------------
    class MockRetriever:
        def invoke(self, question):
            return [type("Doc", (), {"page_content": "content"})()]

    monkeypatch.setattr(
        "src.nodes.retrieve_node.retriever",
        MockRetriever()
    )

    # -----------------------
    # Mock Grader
    # -----------------------
    class MockGrader:
        def invoke(self, payload):
            return {"binary_score": "yes"}

    monkeypatch.setattr(
        "src.nodes.grade_node.retrieval_grader",
        MockGrader()
    )

    # -----------------------
    # Mock Generate Node
    # ⚠️ PATCH WHERE GRAPH IMPORTED IT
    # -----------------------
    def mock_generate(state):
        state["generation"] = "final answer"
        return state

    monkeypatch.setattr(
        "src.langgraphCorrectiveAI.graph.workflow.generate",
        mock_generate
    )

    # -----------------------
    # Build Graph & Invoke
    # -----------------------
    graph = build_graph()
    result = graph.invoke({"question": "test"})

    # -----------------------
    # Assertion
    # -----------------------
    assert result["generation"] == "final answer"
