import pytest
from src.diagnostics import (
    compute_relevance_ratio,
    detect_hallucination_risk,
    evaluate_run,
)

# Relevance Ratio Tests
def test_relevance_ratio_basic():
    assert compute_relevance_ratio(4, 2) == 0.5


def test_relevance_ratio_zero_docs():
    assert compute_relevance_ratio(0, 0) == 0.0

# Hallucination Detection Tests
def test_low_hallucination_risk():
    context = "LangGraph is a framework for building agent workflows."
    answer = "LangGraph is a framework for building agent workflows."

    risk = detect_hallucination_risk(answer, context)

    assert risk < 0.2


def test_high_hallucination_risk():
    context = "LangGraph is a workflow system."
    answer = "LangGraph was invented in 1998 in Germany."

    risk = detect_hallucination_risk(answer, context)

    assert risk > 0.5

# Full Evaluation Test
def test_evaluate_run_structure():
    question = "What is LangGraph?"
    answer = "LangGraph is a workflow orchestration framework."
    docs = ["LangGraph is a workflow orchestration framework."]
    relevant_docs = 1

    result = evaluate_run(
        question=question,
        answer=answer,
        documents=docs,
        relevant_docs_count=relevant_docs,
        used_web_search=False
    )

    assert result.context_relevance_ratio == 1.0
    assert result.retrieved_docs_count == 1
    assert result.used_web_search is False
    assert result.answer_length > 0
