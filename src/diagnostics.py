"""
Diagnostics and evaluation utilities for Corrective RAG.

This module provides:
- Structured evaluation results
- Context quality metrics
- Simple regression-style scoring
- Hallucination detection heuristics
- Reproducible evaluation summaries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import re

# Data Models
@dataclass
class EvaluationResult:
    question: str
    generated_answer: str
    retrieved_docs_count: int
    used_web_search: bool
    context_relevance_ratio: float
    answer_length: int
    hallucination_risk: float

# Core Diagnostics
def compute_relevance_ratio(
    total_docs: int,
    relevant_docs: int
) -> float:
    """
    Computes ratio of relevant documents retrieved.
    """
    if total_docs == 0:
        return 0.0

    return round(relevant_docs / total_docs, 3)


def detect_hallucination_risk(
    answer: str,
    context: str
) -> float:
    """
    Naive heuristic hallucination detector.

    Strategy:
    - Extract key nouns from answer
    - Check how many appear in context
    - Lower overlap → higher hallucination risk

    Returns value between 0.0 (low risk) and 1.0 (high risk)
    """

    if not context.strip():
        return 1.0

    # Basic token extraction (can be improved later)
    answer_tokens = set(re.findall(r"\b[A-Za-z]{4,}\b", answer.lower()))
    context_tokens = set(re.findall(r"\b[A-Za-z]{4,}\b", context.lower()))

    if not answer_tokens:
        return 0.0

    overlap = answer_tokens.intersection(context_tokens)

    overlap_ratio = len(overlap) / len(answer_tokens)

    # Risk is inverse of overlap
    risk = 1 - overlap_ratio

    return round(risk, 3)


def evaluate_run(
    question: str,
    answer: str,
    documents: List[str],
    relevant_docs_count: int,
    used_web_search: bool
) -> EvaluationResult:
    """
    Produces structured evaluation for one graph execution.
    """

    total_docs = len(documents)
    context = "\n".join(documents)

    relevance_ratio = compute_relevance_ratio(
        total_docs=total_docs,
        relevant_docs=relevant_docs_count
    )

    hallucination_score = detect_hallucination_risk(
        answer=answer,
        context=context
    )

    return EvaluationResult(
        question=question,
        generated_answer=answer,
        retrieved_docs_count=total_docs,
        used_web_search=used_web_search,
        context_relevance_ratio=relevance_ratio,
        answer_length=len(answer),
        hallucination_risk=hallucination_score
    )

# Reporting Utilities
def summarize_evaluation(result: EvaluationResult) -> Dict[str, float | int | bool]:
    """
    Converts EvaluationResult into simple summary dictionary.
    Useful for logging / CI / analytics.
    """

    return {
        "retrieved_docs": result.retrieved_docs_count,
        "used_web_search": result.used_web_search,
        "relevance_ratio": result.context_relevance_ratio,
        "answer_length": result.answer_length,
        "hallucination_risk": result.hallucination_risk,
    }
