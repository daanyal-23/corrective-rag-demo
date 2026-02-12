from src.nodes.grade_node import grade_documents

def test_grade_documents_all_relevant(monkeypatch):

    class MockGrader:
        def invoke(self, payload):
            return {"binary_score": "yes"}

    monkeypatch.setattr(
        "src.nodes.grade_node.retrieval_grader",
        MockGrader()
    )

    state = {
        "question": "test",
        "documents": [type("Doc", (), {"page_content": "content"})()]
    }

    result = grade_documents(state)

    assert result["web_search"] == "No"
    assert len(result["documents"]) == 1
