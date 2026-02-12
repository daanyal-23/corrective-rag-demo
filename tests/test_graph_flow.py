from src.langgraphCorrectiveAI.graph.workflow import build_graph

def test_graph_generates_without_web(monkeypatch):

    class MockRetriever:
        def invoke(self, question):
            return [type("Doc", (), {"page_content": "content"})()]

    class MockGrader:
        def invoke(self, payload):
            return {"binary_score": "yes"}

    class MockRagChain:
        def invoke(self, payload):
            return "final answer"

    monkeypatch.setattr(
        "src.nodes.retrieve_node.get_retriever",
        lambda: MockRetriever()
    )

    monkeypatch.setattr(
        "src.nodes.grade_node.get_retrieval_grader",
        lambda: MockGrader()
    )

    monkeypatch.setattr(
        "src.nodes.generate_node.get_rag_chain",
        lambda: MockRagChain()
    )

    graph = build_graph()

    result = graph.invoke({"question": "test"})

    assert result["generation"] == "final answer"
