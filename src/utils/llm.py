"""src/utils/llm.py — single place to build the LLM, works with Groq or OpenAI."""
import os
from src.utils.config import LLM_PROVIDER, LLM_MODEL, GROQ_API_KEY, OPENAI_API_KEY


def get_llm(temperature: float = 0):
    """
    Return a LangChain chat model using whichever provider is configured.
    Groq is used if GROQ_API_KEY is set (free tier), otherwise OpenAI.
    """
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        return ChatGroq(model=LLM_MODEL, temperature=temperature)

    else:  # openai
        from langchain_openai import ChatOpenAI
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


def get_llm_with_structured_output(schema, temperature: float = 0):
    """Return LLM bound to a Pydantic output schema."""
    return get_llm(temperature).with_structured_output(schema)