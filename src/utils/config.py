"""src/utils/config.py — centralised env var loading."""
import os
from dotenv import load_dotenv

load_dotenv()


def require(key: str) -> str:
    """Return env var or raise a clear error if missing."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Missing required env var: {key}\n"
            f"Copy .env.example → .env and fill in your keys."
        )
    return val


# ── Exported constants ────────────────────────────────────────────
OPENAI_API_KEY       = require("OPENAI_API_KEY")
TAVILY_API_KEY       = os.getenv("TAVILY_API_KEY", "")
LANGFUSE_SECRET_KEY  = require("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY  = require("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST        = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

BQ_PROJECT           = os.getenv("BIGQUERY_PROJECT_ID", "")
BQ_DATASET           = os.getenv("BIGQUERY_DATASET", "pe_pipeline")

JUDGE_ACCEPT_THRESHOLD = int(os.getenv("JUDGE_ACCEPT_THRESHOLD", "7"))
JUDGE_REVIEW_THRESHOLD = int(os.getenv("JUDGE_REVIEW_THRESHOLD", "4"))
BATCH_SIZE             = int(os.getenv("BATCH_SIZE", "10"))
