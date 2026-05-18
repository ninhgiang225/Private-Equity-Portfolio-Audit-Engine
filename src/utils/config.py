"""src/utils/config.py — centralised env var loading.

Free-tier setup (no credit card needed):
  LLM      → Groq  : https://console.groq.com   (set GROQ_API_KEY)
  Tracing  → skip  : set LANGFUSE_ENABLED=false  (or get free keys at cloud.langfuse.com)
"""
import os
from dotenv import load_dotenv

load_dotenv()


def optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ── LLM provider — Groq (free) takes priority over OpenAI ────────
GROQ_API_KEY   = optional("GROQ_API_KEY")
OPENAI_API_KEY = optional("OPENAI_API_KEY")

# Which provider to use: auto-detect from available keys
if GROQ_API_KEY:
    LLM_PROVIDER = "groq"
    LLM_MODEL    = optional("LLM_MODEL", "llama3-8b-8192")   # fast free Groq model
elif OPENAI_API_KEY:
    LLM_PROVIDER = "openai"
    LLM_MODEL    = optional("LLM_MODEL", "gpt-4o-mini")
else:
    raise EnvironmentError(
        "No LLM key found. Set either GROQ_API_KEY (free) or OPENAI_API_KEY in your .env\n"
        "  Free option → https://console.groq.com"
    )

# ── Langfuse — fully optional, disabled if keys missing ──────────
LANGFUSE_ENABLED    = optional("LANGFUSE_ENABLED", "true").lower() == "true"
LANGFUSE_SECRET_KEY = optional("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = optional("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST       = optional("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Disable automatically if keys are missing
if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
    LANGFUSE_ENABLED = False

# ── Other services ────────────────────────────────────────────────
TAVILY_API_KEY = optional("TAVILY_API_KEY")
BQ_PROJECT     = optional("BIGQUERY_PROJECT_ID")
BQ_DATASET     = optional("BIGQUERY_DATASET", "pe_pipeline")

JUDGE_ACCEPT_THRESHOLD = int(optional("JUDGE_ACCEPT_THRESHOLD", "7"))
JUDGE_REVIEW_THRESHOLD = int(optional("JUDGE_REVIEW_THRESHOLD", "4"))
BATCH_SIZE             = int(optional("BATCH_SIZE", "10"))