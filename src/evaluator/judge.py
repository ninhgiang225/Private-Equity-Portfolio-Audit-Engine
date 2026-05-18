"""src/evaluator/judge.py

LLM-as-judge that scores each entity resolution match.
Uses a separate, cheaper LLM call (gpt-4o-mini) with structured output
so scores are always parseable — no regex needed.

WEEK 3 milestone:
  - Run evaluate_match() on 20 resolved companies
  - Label those same 20 yourself in the review CSV
  - Compare: does the judge agree with you?
"""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.llm import get_llm_with_structured_output
from src.utils.tracing import get_callback_handler, make_config
from src.utils.config import (
    JUDGE_ACCEPT_THRESHOLD,
    JUDGE_REVIEW_THRESHOLD,
)


# ── Output schema ─────────────────────────────────────────────────────────────
class MatchVerdict(BaseModel):
    """Structured output for the LLM judge."""
    score: int = Field(
        description="Match quality score from 0 to 10. "
                    "10 = perfect obvious match, 0 = completely wrong.",
        ge=0, le=10,
    )
    rationale: str = Field(
        description="One sentence explaining the score.",
        max_length=300,
    )
    verdict: str = Field(
        description="One of: 'accept', 'review', 'reject'",
        pattern="^(accept|review|reject)$",
    )


# ── Prompt ────────────────────────────────────────────────────────────────────
JUDGE_PROMPT = ChatPromptTemplate.from_template("""
You are a data quality judge evaluating whether a company entity was resolved correctly.

Given:
  Raw input:      {raw_name}
  Resolved name:  {resolved_name}
  Domain:         {domain}
  Evidence:       {evidence}
  Confidence:     {confidence}

Score the match quality from 0 to 10:
  10  Perfect match — clearly the same company, no doubt.
  8–9 Very good — strong evidence, minor name variation.
  6–7 Plausible — reasonable match but some uncertainty.
  4–5 Weak — possible match but significant doubt.
  0–3 Wrong — clearly a different company or unresolved.

Return ONLY a JSON with fields: score (int), rationale (str), verdict (str).
Verdict rules: score >= {accept_threshold} → "accept", score < {review_threshold} → "reject", else → "review"
""")


# ── Main function ─────────────────────────────────────────────────────────────
def evaluate_match(
    raw_name: str,
    resolved_name: str | None,
    domain: str | None,
    evidence: str,
    confidence: float,
    trace_id: str | None = None,
) -> MatchVerdict:
    """
    Evaluate a single entity resolution match.

    Args:
        raw_name:      Original messy input, e.g. 'Aple Inc'
        resolved_name: Agent's resolved canonical name, e.g. 'Apple Inc.'
        domain:        Resolved domain, e.g. 'apple.com'
        evidence:      One-sentence evidence string from the resolver.
        confidence:    Agent's self-reported confidence (0.0 – 1.0).
        trace_id:      Optional Langfuse trace ID to link this span.

    Returns:
        MatchVerdict with score, rationale, and verdict.
    """
    if resolved_name is None:
        return MatchVerdict(score=0, rationale="Agent returned no resolved name.", verdict="reject")

    handler = get_callback_handler(trace_id=trace_id, name=f"judge:{raw_name[:40]}")
    llm     = get_llm_with_structured_output(MatchVerdict)
    chain = JUDGE_PROMPT | llm

    verdict: MatchVerdict = chain.invoke(
        {
            "raw_name":         raw_name,
            "resolved_name":    resolved_name,
            "domain":           domain or "unknown",
            "evidence":         evidence,
            "confidence":       confidence,
            "accept_threshold": JUDGE_ACCEPT_THRESHOLD,
            "review_threshold": JUDGE_REVIEW_THRESHOLD,
        },
        config=make_config(handler),
    )
    return verdict


def route_verdict(verdict: MatchVerdict) -> str:
    """Map a MatchVerdict to a pipeline routing decision."""
    return verdict.verdict  # "accept" | "review" | "reject"


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    examples = [
        # (raw, resolved, domain, evidence, confidence)
        ("Aple Inc",   "Apple Inc.",           "apple.com",   "Fuzzy matched score 92/100", 0.95),
        ("MSFT",       "Microsoft Corporation","microsoft.com","Web search confirmed ticker", 0.85),
        ("Stipe Inc",  "Stripe Inc.",           "stripe.com",  "Fuzzy matched score 78/100", 0.7),
        ("WandB",      "Weights & Biases Inc.", "wandb.ai",    "Exact lookup found",         0.9),
        ("PLAID Fintech","Plaid Inc.",          "plaid.com",   "Fuzzy matched score 80/100", 0.8),
    ]

    for raw, resolved, domain, evidence, confidence in examples:
        verdict = evaluate_match(raw, resolved, domain, evidence, confidence)
        print(f"\n{raw!r:20} → {resolved!r}")
        print(f"  Score:    {verdict.score}/10")
        print(f"  Verdict:  {verdict.verdict}")
        print(f"  Reason:   {verdict.rationale}")