"""src/pipeline/nodes.py

Individual node functions for the LangGraph pipeline.
Each function takes and returns the full PipelineState dict.

Node order:  ingest → resolve → evaluate → export
"""
from __future__ import annotations
import pandas as pd
from tqdm import tqdm
from typing import TypedDict

from src.agent.resolver import resolve_company
from src.evaluator.judge import evaluate_match, MatchVerdict
from src.utils.config import BATCH_SIZE


# ── State definition ──────────────────────────────────────────────────────────
class CompanyRecord(TypedDict):
    id: int
    raw_name: str
    canonical_name: str | None
    domain: str | None
    confidence: float
    evidence: str
    judge_score: int | None
    judge_rationale: str | None
    verdict: str | None          # "accept" | "review" | "reject" | "pending"


class PipelineState(TypedDict):
    input_path: str              # path to raw CSV
    records: list[CompanyRecord] # built up as pipeline runs
    accepted: list[CompanyRecord]
    review_queue: list[CompanyRecord]
    rejected: list[CompanyRecord]
    run_id: str                  # Langfuse trace group ID


# ── Node 1: Ingest ────────────────────────────────────────────────────────────
def ingest_node(state: PipelineState) -> PipelineState:
    """
    Read the raw CSV and populate state["records"].
    Only reads id and raw_name — ground_truth columns are ignored.
    """
    df = pd.read_csv(state["input_path"], usecols=["id", "raw_name"])
    state["records"] = [
        CompanyRecord(
            id=int(row.id),
            raw_name=row.raw_name,
            canonical_name=None,
            domain=None,
            confidence=0.0,
            evidence="",
            judge_score=None,
            judge_rationale=None,
            verdict="pending",
        )
        for row in df.itertuples(index=False)
    ]
    print(f"[ingest] Loaded {len(state['records'])} records from {state['input_path']}")
    return state


# ── Node 2: Resolve ───────────────────────────────────────────────────────────
def resolve_node(state: PipelineState) -> PipelineState:
    """
    Run the ReAct entity resolution agent on each record.
    Processes in batches to respect API rate limits.
    """
    records = state["records"]
    run_id = state["run_id"]

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Resolving"):
        batch = records[i : i + BATCH_SIZE]
        for rec in batch:
            result = resolve_company(rec["raw_name"], trace_id=f"{run_id}_{rec['id']}")
            rec["canonical_name"] = result.get("canonical_name")
            rec["domain"]         = result.get("domain")
            rec["confidence"]     = float(result.get("confidence") or 0.0)
            rec["evidence"]       = result.get("evidence", "")

    print(f"[resolve] Resolved {sum(1 for r in records if r['canonical_name'])} / {len(records)}")
    return state


# ── Node 3: Evaluate ─────────────────────────────────────────────────────────
def evaluate_node(state: PipelineState) -> PipelineState:
    """
    Run LLM-as-judge on every resolved record.
    Populates judge_score, judge_rationale, and verdict.
    """
    for rec in tqdm(state["records"], desc="Evaluating"):
        verdict: MatchVerdict = evaluate_match(
            raw_name=rec["raw_name"],
            resolved_name=rec["canonical_name"],
            domain=rec["domain"],
            evidence=rec["evidence"],
            confidence=rec["confidence"],
            trace_id=f"{state['run_id']}_{rec['id']}",
        )
        rec["judge_score"]     = verdict.score
        rec["judge_rationale"] = verdict.rationale
        rec["verdict"]         = verdict.verdict

    accepted = [r for r in state["records"] if r["verdict"] == "accept"]
    review   = [r for r in state["records"] if r["verdict"] == "review"]
    rejected = [r for r in state["records"] if r["verdict"] == "reject"]

    state["accepted"]     = accepted
    state["review_queue"] = review
    state["rejected"]     = rejected

    print(
        f"[evaluate] accept={len(accepted)}  review={len(review)}  reject={len(rejected)}"
    )
    return state


# ── Node 4: Export ────────────────────────────────────────────────────────────
def export_node(state: PipelineState) -> PipelineState:
    """
    Write results to CSV (and optionally BigQuery).
    - data/resolved_output.csv  — all records with verdicts
    - data/review_queue.csv     — records needing human review
    """
    all_df    = pd.DataFrame(state["records"])
    review_df = pd.DataFrame(state["review_queue"])

    all_df.to_csv("../data/resolved_output.csv", index=False)
    print(f"[export] Wrote {len(all_df)} rows → data/resolved_output.csv")

    if not review_df.empty:
        review_df.to_csv("../data/review_queue.csv", index=False)
        print(f"[export] Wrote {len(review_df)} rows → data/review_queue.csv (needs human review)")

    # ── Optional BigQuery write (uncomment in Week 4) ──────────────────────
    # from src.utils.bigquery import write_dataframe
    # write_dataframe(all_df, table="resolved_companies")

    return state
