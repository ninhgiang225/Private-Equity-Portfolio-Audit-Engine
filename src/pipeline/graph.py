"""src/pipeline/graph.py

LangGraph StateGraph wiring ingest → resolve → evaluate → export.
The conditional edge after evaluate() routes to a "needs_review" sink
so you can inspect the graph visually in LangGraph Studio.

WEEK 4 milestone: run this file and confirm all 100 companies are processed.

Usage:
    python -m src.pipeline.graph
    python -m src.pipeline.graph --input data/raw_companies.csv --limit 10
"""
from __future__ import annotations
import argparse
import uuid

from langgraph.graph import StateGraph, END

from src.pipeline.nodes import (
    PipelineState,
    ingest_node,
    resolve_node,
    evaluate_node,
    export_node,
)


# ── Conditional routing ───────────────────────────────────────────────────────
def all_resolved(state: PipelineState) -> str:
    """
    After evaluate: if there is a review queue, flag it; always continue to export.
    Extend this to pause and wait for human input if needed.
    """
    if state.get("review_queue"):
        print(
            f"\n⚠️  {len(state['review_queue'])} records need human review → data/review_queue.csv"
        )
    return "export"   # always go to export in this version


# ── Build the graph ───────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("ingest",   ingest_node)
    g.add_node("resolve",  resolve_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("export",   export_node)

    g.set_entry_point("ingest")

    g.add_edge("ingest",  "resolve")
    g.add_edge("resolve", "evaluate")

    # Conditional edge — routes to "export" (or could pause for review)
    g.add_conditional_edges(
        "evaluate",
        all_resolved,
        {"export": "export"},
    )

    g.add_edge("export", END)
    return g


# ── Entry point ───────────────────────────────────────────────────────────────
def run_pipeline(input_path: str = "../data/raw_companies.csv", limit: int | None = None) -> PipelineState:
    import pandas as pd

    graph    = build_graph().compile()
    run_id   = str(uuid.uuid4())[:8]

    initial_state: PipelineState = {
        "input_path":  input_path,
        "records":     [],
        "accepted":    [],
        "review_queue":[],
        "rejected":    [],
        "run_id":      run_id,
    }

    # Optional row limit for quick dev runs
    if limit:
        df = pd.read_csv(input_path).head(limit)
        tmp_path = f"../data/_tmp_limit_{limit}.csv"
        df.to_csv(tmp_path, index=False)
        initial_state["input_path"] = tmp_path

    print(f"\nPipeline run {run_id} starting...\n")
    final_state = graph.invoke(initial_state)

    print("\n── Run summary ──────────────────────────────────")
    print(f"  Total:    {len(final_state['records'])}")
    print(f"  Accepted: {len(final_state['accepted'])}")
    print(f"  Review:   {len(final_state['review_queue'])}")
    print(f"  Rejected: {len(final_state['rejected'])}")
    print(f"  Run ID:   {run_id}  (search in Langfuse)")
    print("─────────────────────────────────────────────────\n")

    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PE entity resolution pipeline")
    parser.add_argument("--input", default="../data/raw_companies.csv")
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows (dev mode)")
    args = parser.parse_args()

    run_pipeline(input_path=args.input, limit=args.limit)
