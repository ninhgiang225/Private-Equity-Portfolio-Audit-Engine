"""src/reporting/charts.py

Generate a self-contained HTML report of a pipeline run.
No Streamlit needed — just open the output file in any browser.

Usage:
    python -m src.reporting.charts                         # uses data/resolved_output.csv
    python -m src.reporting.charts --input path/to/run.csv
"""
from __future__ import annotations
import argparse
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


VERDICT_COLORS = {
    "accept":  "#2d6a4f",
    "review":  "#92600a",
    "reject":  "#c84b2f",
    "pending": "#b8b2aa",
}

THEME = dict(
    font_family="Georgia, serif",
    paper_bgcolor="#f7f4ef",
    plot_bgcolor="rgba(238,234,227,0.4)",
)


def make_demo_data(n: int = 100) -> pd.DataFrame:
    import numpy as np
    np.random.seed(42)
    verdicts = np.random.choice(["accept", "review", "reject"], size=n, p=[0.62, 0.25, 0.13])
    scores   = [np.random.randint(7, 11) if v == "accept"
                else np.random.randint(4, 7) if v == "review"
                else np.random.randint(0, 4) for v in verdicts]
    return pd.DataFrame({
        "id":           range(1, n + 1),
        "raw_name":     [f"Company {i}" for i in range(n)],
        "canonical_name": [f"Resolved {i}" for i in range(n)],
        "confidence":   np.round(np.random.uniform(0.5, 0.99, n), 2),
        "judge_score":  scores,
        "verdict":      verdicts,
        "difficulty":   np.random.choice(["easy", "medium", "hard"], n),
    })


def build_report(df: pd.DataFrame, output_path: str = "data/pipeline_report.html") -> str:
    total    = len(df)
    accepted = (df["verdict"] == "accept").sum()
    review   = (df["verdict"] == "review").sum()
    rejected = (df["verdict"] == "reject").sum()
    avg_score = round(df["judge_score"].mean(), 1) if "judge_score" in df.columns else 0

    # ── Fig 1: Score histogram ─────────────────────────────────────────────
    fig1 = px.histogram(
        df, x="judge_score", nbins=11,
        color_discrete_sequence=["#2a5a8c"],
        title="Judge Score Distribution",
        labels={"judge_score": "Score (0–10)"},
    )
    fig1.update_layout(**THEME, height=300,
                       xaxis=dict(tickmode="linear", tick0=0, dtick=1),
                       margin=dict(l=40, r=20, t=50, b=40))
    fig1.add_vrect(x0=-0.5, x1=3.5,  fillcolor="#c84b2f", opacity=0.07, line_width=0)
    fig1.add_vrect(x0=3.5,  x1=6.5,  fillcolor="#92600a", opacity=0.07, line_width=0)
    fig1.add_vrect(x0=6.5,  x1=10.5, fillcolor="#2d6a4f", opacity=0.07, line_width=0)

    # ── Fig 2: Verdict donut ───────────────────────────────────────────────
    vc = df["verdict"].value_counts().reset_index()
    vc.columns = ["verdict", "count"]
    fig2 = px.pie(vc, names="verdict", values="count",
                  color="verdict", color_discrete_map=VERDICT_COLORS,
                  title="Verdict Breakdown", hole=0.55)
    fig2.update_traces(textposition="outside", textinfo="percent+label")
    fig2.update_layout(**THEME, height=300, margin=dict(l=20, r=20, t=50, b=20),
                       showlegend=False)

    # ── Fig 3: Difficulty stacked bar ──────────────────────────────────────
    if "difficulty" in df.columns:
        grp = df.groupby(["difficulty", "verdict"]).size().reset_index(name="count")
        fig3 = px.bar(grp, x="difficulty", y="count", color="verdict",
                      color_discrete_map=VERDICT_COLORS,
                      category_orders={"difficulty": ["easy", "medium", "hard"]},
                      title="Resolution by Difficulty", barmode="stack",
                      labels={"difficulty": "", "count": "Companies"})
        fig3.update_layout(**THEME, height=300, margin=dict(l=40, r=20, t=50, b=40),
                           legend=dict(orientation="h", y=-0.2))
    else:
        fig3 = go.Figure()

    # ── Fig 4: Confidence vs Score scatter ─────────────────────────────────
    if "confidence" in df.columns:
        fig4 = px.scatter(df, x="confidence", y="judge_score",
                          color="verdict", color_discrete_map=VERDICT_COLORS,
                          opacity=0.7, title="Confidence vs Judge Score",
                          labels={"confidence": "Agent Confidence", "judge_score": "Judge Score"},
                          hover_data=["raw_name"] if "raw_name" in df.columns else None)
        fig4.add_hline(y=7, line_dash="dot", line_color="#2d6a4f",
                       annotation_text="accept ≥7", annotation_font_size=10)
        fig4.add_hline(y=4, line_dash="dot", line_color="#c84b2f",
                       annotation_text="reject <4", annotation_font_size=10)
        fig4.update_layout(**THEME, height=320, margin=dict(l=40, r=20, t=50, b=40),
                           legend=dict(orientation="h", y=-0.18),
                           yaxis=dict(range=[-0.5, 10.5], tickmode="linear", dtick=1))
    else:
        fig4 = go.Figure()

    # ── Assemble HTML ──────────────────────────────────────────────────────
    def fig_html(fig: go.Figure) -> str:
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PE Pipeline Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,700;1,300&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'DM Sans', sans-serif; background: #f7f4ef; color: #0f0e0d; padding: 2rem; }}
  h1 {{ font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #7a746e; font-size: 14px; margin-bottom: 2rem; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-bottom: 2rem; }}
  .kpi {{ background: white; border: 1px solid rgba(15,14,13,0.1); border-radius: 10px;
          padding: 1.25rem; text-align: center; }}
  .kpi-label {{ font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 0.12em;
                text-transform: uppercase; color: #7a746e; margin-bottom: 0.4rem; }}
  .kpi-value {{ font-size: 2rem; font-weight: 500; line-height: 1; }}
  .kpi-sub {{ font-size: 11px; color: #b8b2aa; margin-top: 0.3rem; }}
  .charts-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }}
  .chart-card {{ background: white; border: 1px solid rgba(15,14,13,0.1); border-radius: 10px;
                 padding: 1rem; }}
  .chart-full {{ background: white; border: 1px solid rgba(15,14,13,0.1); border-radius: 10px;
                 padding: 1rem; margin-bottom: 1rem; }}
  .label {{ font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 0.12em;
            text-transform: uppercase; color: #c84b2f; margin-bottom: 0.5rem; }}
  footer {{ font-family: 'DM Mono', monospace; font-size: 11px; color: #b8b2aa;
            text-align: center; margin-top: 2rem; padding-top: 1rem;
            border-top: 1px solid rgba(15,14,13,0.1); }}
</style>
</head>
<body>

<h1>PE Portfolio Intelligence Pipeline</h1>
<p class="subtitle">Entity Resolution Report &nbsp;·&nbsp; LangChain + LangGraph + Langfuse</p>

<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Total</div>
    <div class="kpi-value">{total}</div>
    <div class="kpi-sub">companies</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Accepted</div>
    <div class="kpi-value" style="color:#2d6a4f">{accepted}</div>
    <div class="kpi-sub">{accepted/total*100:.0f}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Review</div>
    <div class="kpi-value" style="color:#92600a">{review}</div>
    <div class="kpi-sub">needs human</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Rejected</div>
    <div class="kpi-value" style="color:#c84b2f">{rejected}</div>
    <div class="kpi-sub">unresolved</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg Score</div>
    <div class="kpi-value">{avg_score}</div>
    <div class="kpi-sub">out of 10</div>
  </div>
</div>

<div class="charts-2">
  <div class="chart-card">
    <div class="label">Judge Score Distribution</div>
    {fig_html(fig1)}
  </div>
  <div class="chart-card">
    <div class="label">Verdict Breakdown</div>
    {fig_html(fig2)}
  </div>
</div>

<div class="charts-2">
  <div class="chart-card">
    <div class="label">Resolution by Difficulty</div>
    {fig_html(fig3)}
  </div>
  <div class="chart-card">
    <div class="label">Confidence vs Judge Score</div>
    {fig_html(fig4)}
  </div>
</div>

<footer>PE Portfolio Intelligence Pipeline &nbsp;·&nbsp; Auto-generated report</footer>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"[report] Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/resolved_output.csv")
    parser.add_argument("--output", default="data/pipeline_report.html")
    args = parser.parse_args()

    if os.path.exists(args.input):
        df = pd.read_csv(args.input)
        print(f"[report] Loaded {len(df)} rows from {args.input}")
    else:
        print("[report] No results found — using demo data")
        df = make_demo_data()

    build_report(df, args.output)
    print(f"[report] Open in browser: file://{os.path.abspath(args.output)}")
