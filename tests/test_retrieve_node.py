def test_retrieve_with_mock(monkeypatch):

    class MockRetriever:
        def invoke(self, question):
            return ["doc1", "doc2"]

    monkeypatch.setattr(
        "src.nodes.retrieve_node.get_retriever",
        lambda: MockRetriever()
    )

    from src.nodes.retrieve_node import retrive

    state = {"question": "test"}
    result = retrive(state)

    assert result["documents"] == ["doc1", "doc2"]
