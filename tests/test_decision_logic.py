from src.langgraphCorrectiveAI.graph.workflow import decide_to_generate

def test_decide_to_generate_when_relevant():
    state = {
        "web_search": "No",
        "documents": ["doc1"]
    }

    result = decide_to_generate(state)

    assert result == "generate"


def test_decide_to_generate_when_not_relevant():
    state = {
        "web_search": "Yes",
        "documents": []
    }

    result = decide_to_generate(state)

    assert result == "transform_query"
