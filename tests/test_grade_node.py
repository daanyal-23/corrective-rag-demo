def test_grade_documents_all_relevant(monkeypatch):

    class MockGrader:
        def invoke(self, payload):
            return {"binary_score": "yes"}

    monkeypatch.setattr(
        "src.nodes.grade_node.get_retrieval_grader",
        lambda: MockGrader()
    )

    from src.nodes.grade_node import grade_documents

    docs = [type("Doc", (), {"page_content": "content"})()]

    state = {"question": "test", "documents": docs}
    result = grade_documents(state)

    assert len(result["documents"]) == 1
    assert result["web_search"] == "No"
