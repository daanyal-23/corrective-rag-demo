import types
from src.nodes.retrieve_node import retrive

def test_retrieve_with_mock(monkeypatch):

    class MockRetriever:
        def invoke(self, question):
            return ["doc1", "doc2"]

    monkeypatch.setattr(
        "src.nodes.retrieve_node.retriever",
        MockRetriever()
    )

    state = {"question": "test"}

    result = retrive(state)

    assert "documents" in result
    assert len(result["documents"]) == 2
