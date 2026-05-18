"""src/dashboard/app.py

Streamlit dashboard for the PE Portfolio Intelligence Pipeline.
Shows pipeline run results, judge score distributions, and resolution stats.

Run with:
    streamlit run src/dashboard/app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PE Pipeline Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #f7f4ef; }

    .metric-card {
        background: white;
        border: 1px solid rgba(15,14,13,0.1);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .metric-label {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7a746e;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 500;
        color: #0f0e0d;
        line-height: 1;
    }
    .metric-sub {
        font-size: 12px;
        color: #b8b2aa;
        margin-top: 0.3rem;
    }
    .verdict-accept { color: #2d6a4f; }
    .verdict-review { color: #92600a; }
    .verdict-reject { color: #c84b2f; }

    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #c84b2f;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
VERDICT_COLORS = {
    "accept": "#2d6a4f",
    "review": "#92600a",
    "reject": "#c84b2f",
    "pending": "#b8b2aa",
}

PLOTLY_THEME = dict(
    font_family="DM Sans, sans-serif",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(247,244,239,0.5)",
)


def load_data(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_ground_truth() -> pd.DataFrame | None:
    return load_data("data/raw_companies.csv")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏢 PE Pipeline")
    st.markdown("---")

    results_path = st.text_input("Results CSV", value="data/resolved_output.csv")
    df = load_data(results_path)

    st.markdown("---")
    st.markdown("""
    <div style='font-family: DM Mono, monospace; font-size: 11px; color: #7a746e; line-height: 1.8;'>
    <b>PIPELINE STAGES</b><br>
    1. Ingest raw CSV<br>
    2. ReAct agent resolves<br>
    3. LLM-as-judge scores<br>
    4. Export to BigQuery
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if df is not None:
        run_id = df.get("run_id", pd.Series(["unknown"])).iloc[0] if "run_id" in df.columns else "—"
        st.markdown(f"""
        <div style='font-family: DM Mono, monospace; font-size: 11px; color: #b8b2aa;'>
        Run ID: {run_id}
        </div>
        """, unsafe_allow_html=True)


# ── Demo data (used when no real run exists yet) ──────────────────────────────
def make_demo_data() -> pd.DataFrame:
    import numpy as np
    np.random.seed(42)
    gt = load_ground_truth()
    n = len(gt) if gt is not None else 100

    verdicts = np.random.choice(
        ["accept", "review", "reject"],
        size=n,
        p=[0.62, 0.25, 0.13],
    )
    scores = []
    for v in verdicts:
        if v == "accept":  scores.append(np.random.randint(7, 11))
        elif v == "review": scores.append(np.random.randint(4, 7))
        else:               scores.append(np.random.randint(0, 4))

    difficulties = []
    if gt is not None and "difficulty" in gt.columns:
        difficulties = gt["difficulty"].tolist()
    else:
        difficulties = np.random.choice(["easy", "medium", "hard"], size=n).tolist()

    raw_names = gt["raw_name"].tolist() if gt is not None else [f"Company {i}" for i in range(n)]
    resolved  = gt["ground_truth_name"].tolist() if gt is not None else [f"Resolved Co {i}" for i in range(n)]

    return pd.DataFrame({
        "id":             range(1, n + 1),
        "raw_name":       raw_names,
        "canonical_name": resolved,
        "confidence":     np.round(np.random.uniform(0.55, 0.99, n), 2),
        "judge_score":    scores,
        "verdict":        verdicts,
        "difficulty":     difficulties,
        "evidence":       [f"Fuzzy matched with score {np.random.randint(70,99)}/100"] * n,
    })


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family: DM Sans, sans-serif; font-size: 2rem; font-weight: 500;
           color: #0f0e0d; margin-bottom: 0.2rem;'>
  PE Portfolio Intelligence Pipeline
</h1>
<p style='color: #7a746e; font-size: 14px; margin-bottom: 2rem;'>
  Entity resolution · LLM-as-judge · Langfuse observability
</p>
""", unsafe_allow_html=True)

if df is None:
    st.info("📂 No results file found — showing **demo data**. Run the pipeline first: `python -m src.pipeline.graph`")
    df = make_demo_data()

# ── KPI row ───────────────────────────────────────────────────────────────────
total    = len(df)
accepted = (df["verdict"] == "accept").sum()
review   = (df["verdict"] == "review").sum()
rejected = (df["verdict"] == "reject").sum()
avg_score = df["judge_score"].mean() if "judge_score" in df.columns else 0
resolved_pct = (df["canonical_name"].notna().sum() / total * 100) if total else 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total</div>
        <div class="metric-value">{total}</div>
        <div class="metric-sub">companies processed</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Accepted</div>
        <div class="metric-value verdict-accept">{accepted}</div>
        <div class="metric-sub">{accepted/total*100:.0f}% of total</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">For Review</div>
        <div class="metric-value verdict-review">{review}</div>
        <div class="metric-sub">needs human check</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rejected</div>
        <div class="metric-value verdict-reject">{rejected}</div>
        <div class="metric-sub">unresolved</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Judge Score</div>
        <div class="metric-value">{avg_score:.1f}</div>
        <div class="metric-sub">out of 10</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.2, 1.2, 1])

with c1:
    st.markdown('<div class="section-label">Judge Score Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(
        df, x="judge_score", nbins=11,
        color_discrete_sequence=["#2a5a8c"],
        labels={"judge_score": "Score (0–10)", "count": "# Companies"},
    )
    fig.update_layout(
        **PLOTLY_THEME,
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        bargap=0.15,
        xaxis=dict(tickmode="linear", tick0=0, dtick=1, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
    )
    # Colour bars by verdict zone
    fig.add_vrect(x0=-0.5, x1=3.5,  fillcolor="#c84b2f", opacity=0.08, line_width=0)
    fig.add_vrect(x0=3.5,  x1=6.5,  fillcolor="#92600a", opacity=0.08, line_width=0)
    fig.add_vrect(x0=6.5,  x1=10.5, fillcolor="#2d6a4f", opacity=0.08, line_width=0)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with c2:
    st.markdown('<div class="section-label">Resolution by Difficulty</div>', unsafe_allow_html=True)
    if "difficulty" in df.columns:
        grp = df.groupby(["difficulty", "verdict"]).size().reset_index(name="count")
        fig2 = px.bar(
            grp, x="difficulty", y="count", color="verdict",
            color_discrete_map=VERDICT_COLORS,
            category_orders={"difficulty": ["easy", "medium", "hard"]},
            labels={"difficulty": "", "count": ""},
            barmode="stack",
        )
        fig2.update_layout(
            **PLOTLY_THEME,
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=-0.15, x=0, font_size=11),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

with c3:
    st.markdown('<div class="section-label">Verdict Breakdown</div>', unsafe_allow_html=True)
    verdict_counts = df["verdict"].value_counts().reset_index()
    verdict_counts.columns = ["verdict", "count"]
    fig3 = px.pie(
        verdict_counts, names="verdict", values="count",
        color="verdict",
        color_discrete_map=VERDICT_COLORS,
        hole=0.55,
    )
    fig3.update_traces(textposition="outside", textinfo="percent+label", textfont_size=11)
    fig3.update_layout(
        **PLOTLY_THEME,
        height=280,
        margin=dict(l=20, r=20, t=10, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

# ── Confidence vs Judge Score scatter ─────────────────────────────────────────
st.markdown('<div class="section-label">Confidence vs Judge Score</div>', unsafe_allow_html=True)
if "confidence" in df.columns and "judge_score" in df.columns:
    fig4 = px.scatter(
        df, x="confidence", y="judge_score",
        color="verdict",
        color_discrete_map=VERDICT_COLORS,
        hover_data=["raw_name", "canonical_name"],
        labels={"confidence": "Agent Confidence", "judge_score": "Judge Score"},
        opacity=0.75,
        size_max=8,
    )
    fig4.update_traces(marker=dict(size=7))
    fig4.add_hline(y=7, line_dash="dot", line_color="#2d6a4f",  annotation_text="accept threshold", annotation_font_size=10)
    fig4.add_hline(y=4, line_dash="dot", line_color="#c84b2f",  annotation_text="reject threshold", annotation_font_size=10)
    fig4.update_layout(
        **PLOTLY_THEME,
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=-0.15, font_size=11),
        xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)", range=[-0.5, 10.5], tickmode="linear", dtick=1),
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

# ── Results table ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Full Results</div>', unsafe_allow_html=True)

verdict_filter = st.multiselect(
    "Filter by verdict",
    options=["accept", "review", "reject"],
    default=["accept", "review", "reject"],
)

display_cols = [c for c in ["id", "raw_name", "canonical_name", "confidence", "judge_score", "verdict", "evidence"] if c in df.columns]
filtered = df[df["verdict"].isin(verdict_filter)][display_cols]

def color_verdict(val):
    colors = {"accept": "#e8f4ee", "review": "#fdf3e0", "reject": "#fde8e4"}
    return f"background-color: {colors.get(val, 'white')}"

styled = filtered.style.applymap(color_verdict, subset=["verdict"])
st.dataframe(styled, use_container_width=True, height=400)

st.caption(f"Showing {len(filtered)} of {total} records")

# ── Accuracy check (if ground truth available) ────────────────────────────────
gt = load_ground_truth()
if gt is not None and "ground_truth_name" in gt.columns and "canonical_name" in df.columns:
    st.markdown("---")
    st.markdown('<div class="section-label">Accuracy vs Ground Truth</div>', unsafe_allow_html=True)

    merged = df.merge(gt[["id", "ground_truth_name", "difficulty"]], on="id", how="left")
    merged["correct"] = merged["canonical_name"].str.strip().str.lower() == merged["ground_truth_name"].str.strip().str.lower()

    overall_acc = merged["correct"].mean() * 100
    acc_by_diff = merged.groupby("difficulty")["correct"].mean() * 100

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Overall Accuracy", f"{overall_acc:.1f}%")
    for col, diff in zip([col_b, col_c, col_d], ["easy", "medium", "hard"]):
        with col:
            val = acc_by_diff.get(diff, 0)
            st.metric(f"{diff.capitalize()} Accuracy", f"{val:.1f}%")
