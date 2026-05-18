# PE Portfolio Intelligence Pipeline

A LangChain + LangGraph pipeline that ingests messy company name lists,
resolves entities to their canonical form, evaluates match quality with
LLM-as-judge, and traces everything end-to-end via Langfuse.

## Project structure

```
pe_pipeline/
├── data/
│   ├── raw_companies.csv           ← 100 messy company names (your input)
│   ├── resolved_output.csv         ← generated after a pipeline run
│   ├── review_queue.csv            ← low-confidence matches needing human review
│   └── pipeline_report.html        ← auto-generated visual report (open in browser)
├── src/
│   ├── agent/
│   │   ├── tools.py                ← fuzzy_match, web_search, company_lookup tools
│   │   └── resolver.py             ← ReAct agent with robust JSON output parsing
│   ├── evaluator/
│   │   └── judge.py                ← LLM-as-judge with Pydantic structured output
│   ├── pipeline/
│   │   ├── graph.py                ← LangGraph StateGraph wiring all nodes
│   │   └── nodes.py                ← ingest / resolve / evaluate / export nodes
│   ├── dashboard/
│   │   └── app.py                  ← Streamlit live dashboard (run after pipeline)
│   ├── reporting/
│   │   └── charts.py               ← standalone HTML report generator (no Streamlit needed)
│   └── utils/
│       ├── config.py               ← env var loading; auto-detects Groq vs OpenAI
│       ├── llm.py                  ← LLM factory (swap provider in one place)
│       ├── tracing.py              ← safe Langfuse wrapper; no-ops if keys missing
│       └── bigquery.py             ← BigQuery read/write helpers (Week 4)
├── notebooks/
│   └── 01_exploration.ipynb.py     ← sandbox for testing individual pieces
├── tests/
│   ├── test_tools.py               ← unit tests for fuzzy_match & company_lookup
│   └── test_judge.py               ← integration tests for LLM judge
├── .env.example                    ← copy to .env and fill in keys
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Clone and create a virtual env
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env file and fill in your keys
cp .env.example .env

# 4. Run the full pipeline
python -m src.pipeline.graph
```

## Four learning milestones

| Week | Focus | Done when... |
|------|-------|-------------|
| 1 | Setup + first LangChain trace | Langfuse dashboard shows a trace |
| 2 | Entity resolution agent | Agent resolves 10 test companies |
| 3 | LLM-as-judge + routing | Review CSV generated, scores logged |
| 4 | LangGraph + full batch run | All 100 companies resolved & written to BigQuery |

## Key concepts practiced

- LangChain ReAct agent with custom tools
- LangGraph `StateGraph` with conditional edges
- LLM-as-judge with `with_structured_output()` + Pydantic
- Langfuse `CallbackHandler` for tracing every LLM call
- Human-in-the-loop review queue pattern