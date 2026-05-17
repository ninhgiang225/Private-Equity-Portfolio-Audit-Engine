"""src/agent/resolver.py

A LangChain ReAct agent that resolves a messy company name to its
canonical form using fuzzy matching, database lookup, and web search.

WEEK 2 milestone: run resolve_company() on 10 test names and verify output.
"""
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langfuse.callback import CallbackHandler

from src.agent.tools import ALL_TOOLS
from src.utils.config import OPENAI_API_KEY

import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ── System prompt ─────────────────────────────────────────────────────────────
RESOLVER_PROMPT = PromptTemplate.from_template("""
You are an expert data analyst specializing in company entity resolution.
Your goal is to identify the canonical (official, full legal) name of a company
given a potentially messy, abbreviated, or misspelled input.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Strategy:
1. Try company_lookup_tool first (exact match, free).
2. If no exact match, try fuzzy_match_tool (handles typos).
3. If still no match (score < 75), use web_search_tool.
4. Once you have a canonical name, stop — do NOT keep searching.

Always end with a Final Answer in this exact JSON format:
{{
  "canonical_name": "Official Company Name Inc.",
  "domain": "company.com",
  "confidence": 0.9,
  "evidence": "One sentence explaining how you resolved it."
}}

If you cannot resolve the company after using all tools, return:
{{
  "canonical_name": null,
  "domain": null,
  "confidence": 0.0,
  "evidence": "Could not resolve — requires manual review."
}}

Begin!

Company to resolve: {input}

{agent_scratchpad}
""")


def build_resolver_agent() -> AgentExecutor:
    """Build and return a configured ReAct agent executor."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=RESOLVER_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,          # set False in production
        max_iterations=5,      # prevent infinite loops
        handle_parsing_errors=True,
    )


def resolve_company(raw_name: str, trace_id: str | None = None) -> dict:
    """
    Resolve a single company name.

    Args:
        raw_name:  The messy input string, e.g. 'Aple Inc'
        trace_id:  Optional Langfuse trace ID for linking spans.

    Returns:
        dict with canonical_name, domain, confidence, evidence.
    """
    langfuse_handler = CallbackHandler(
        trace_id=trace_id,
        name=f"resolve:{raw_name[:40]}",
    )

    executor = build_resolver_agent()

    try:
        result = executor.invoke(
            {"input": raw_name},
            config={"callbacks": [langfuse_handler]},
        )
        output_text = result.get("output", "{}")

        # Parse the JSON block from the agent's Final Answer
        import json, re
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if match:
            return json.loads(match.group())

        # Fallback if the agent returned plain text
        return {
            "canonical_name": None,
            "domain": None,
            "confidence": 0.0,
            "evidence": output_text,
        }

    except Exception as e:
        return {
            "canonical_name": None,
            "domain": None,
            "confidence": 0.0,
            "evidence": f"Agent error: {str(e)}",
        }


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = ["Aple Inc", "MSFT", "Saleforce.com", "Figma Design", "WandB"]
    for name in test_cases:
        print(f"\n{'─'*50}")
        print(f"Input:  {name}")
        result = resolve_company(name)
        print(f"Output: {result}")
