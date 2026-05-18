"""tests/test_judge.py

Integration tests for the LLM-as-judge evaluator.
These call the OpenAI API — set OPENAI_API_KEY in .env before running.

Run with:  pytest tests/test_judge.py -v
"""
import pytest
from src.evaluator.judge import evaluate_match, MatchVerdict
from src.utils.config import JUDGE_ACCEPT_THRESHOLD, JUDGE_REVIEW_THRESHOLD


@pytest.fixture
def obvious_match():
    return dict(
        raw_name="Aple Inc",
        resolved_name="Apple Inc.",
        domain="apple.com",
        evidence="Fuzzy matched with score 92/100",
        confidence=0.95,
    )

@pytest.fixture
def obvious_mismatch():
    return dict(
        raw_name="Stipe Inc",
        resolved_name="Oracle Corporation",   # clearly wrong
        domain="oracle.com",
        evidence="Exact lookup found",
        confidence=0.9,
    )


class TestJudgeOutputSchema:
    def test_returns_match_verdict(self, obvious_match):
        result = evaluate_match(**obvious_match)
        assert isinstance(result, MatchVerdict)

    def test_score_in_range(self, obvious_match):
        result = evaluate_match(**obvious_match)
        assert 0 <= result.score <= 10

    def test_verdict_is_valid(self, obvious_match):
        result = evaluate_match(**obvious_match)
        assert result.verdict in ("accept", "review", "reject")

    def test_rationale_is_string(self, obvious_match):
        result = evaluate_match(**obvious_match)
        assert isinstance(result.rationale, str) and len(result.rationale) > 0


class TestJudgeRouting:
    def test_good_match_is_accepted(self, obvious_match):
        result = evaluate_match(**obvious_match)
        # A clear typo→correct-name match should score high
        assert result.score >= JUDGE_REVIEW_THRESHOLD

    def test_null_resolution_is_rejected(self):
        result = evaluate_match(
            raw_name="XYZXYZ",
            resolved_name=None,
            domain=None,
            evidence="Could not resolve",
            confidence=0.0,
        )
        assert result.verdict == "reject"
        assert result.score == 0
