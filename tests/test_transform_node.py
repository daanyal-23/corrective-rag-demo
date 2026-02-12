def test_transform_query(monkeypatch):

    class MockRewriter:
        def invoke(self, payload):
            return "better question"

    monkeypatch.setattr(
        "src.nodes.transform_node.get_question_rewriter",
        lambda: MockRewriter()
    )

    from src.nodes.transform_node import transform_query

    state = {"question": "test", "documents": []}
    result = transform_query(state)

    assert result["question"] == "better question"
