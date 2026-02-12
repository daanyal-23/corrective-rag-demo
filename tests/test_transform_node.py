from src.nodes.transform_node import transform_query

def test_transform_query(monkeypatch):

    class MockRewriter:
        def invoke(self, payload):
            return "better question"

    monkeypatch.setattr(
        "src.nodes.transform_node.question_rewriter",
        MockRewriter()
    )

    state = {
        "question": "bad question",
        "documents": []
    }

    result = transform_query(state)

    assert result["question"] == "better question"
