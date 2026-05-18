"""src/utils/tracing.py — safe Langfuse wrapper that no-ops when disabled."""
from src.utils.config import (
    LANGFUSE_ENABLED,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_HOST,
)
import os


def get_callback_handler(trace_id: str | None = None, name: str = ""):
    """
    Return a Langfuse CallbackHandler if tracing is enabled,
    otherwise return None (LangChain safely ignores None in callbacks list).
    """
    if not LANGFUSE_ENABLED:
        return None

    os.environ.setdefault("LANGFUSE_SECRET_KEY", LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_HOST",       LANGFUSE_HOST)

    try:
        try:
            from langfuse.langchain import CallbackHandler   # langfuse >= 2.0
        except ImportError:
            from langfuse.callback import CallbackHandler    # langfuse < 2.0

        return CallbackHandler()
    except Exception as e:
        print(f"[tracing] Langfuse init failed ({e}) — continuing without tracing")
        return None


def make_config(handler) -> dict:
    """Build a LangChain invoke config, skipping None callbacks."""
    callbacks = [h for h in [handler] if h is not None]
    return {"callbacks": callbacks} if callbacks else {}