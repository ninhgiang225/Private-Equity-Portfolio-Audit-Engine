"""src/agent/resolver.py

A LangChain ReAct agent that resolves a messy company name to its
canonical form using fuzzy matching, database lookup, and web search.

WEEK 2 milestone: run resolve_company() on 10 test names and verify output.
"""
from langchain_core.prompts import PromptTemplate

# LangGraph 0.2+ unified the ReAct agent — try modern import first
try:
    from langgraph.prebuilt import create_react_agent
    _USE_LANGGRAPH = True
except ImportError:
    from langchain.agents import create_react_agent, AgentExecutor
    _USE_LANGGRAPH = False

from src.agent.tools import ALL_TOOLS
from src.utils.llm import get_llm
from src.utils.tracing import get_callback_handler, make_config


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
{
  "canonical_name": "Official Company Name Inc.",
  "domain": "company.com",
  "confidence": 0.9,
  "evidence": "One sentence explaining how you resolved it."
}

If you cannot resolve the company after using all tools, return:
{
  "canonical_name": null,
  "domain": null,
  "confidence": 0.0,
  "evidence": "Could not resolve — requires manual review."
}

Begin!

Company to resolve: {input}

{agent_scratchpad}
""")


def build_resolver_agent():
    """Build and return a configured agent — works with LangGraph >= 0.2 and older LangChain."""
    llm = get_llm(temperature=0)

    if _USE_LANGGRAPH:
        # Modern API: create_react_agent takes (model, tools, prompt)
        # Returns a CompiledGraph — call .invoke() directly, no AgentExecutor needed
        from langchain_core.messages import SystemMessage
        system_msg = SystemMessage(content=RESOLVER_PROMPT.template.split("{input}")[0].strip())
        return create_react_agent(llm, tools=ALL_TOOLS, prompt=system_msg)
    else:
        # Legacy API: needs explicit AgentExecutor wrapper
        agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=RESOLVER_PROMPT)
        return AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=True,
            max_iterations=5,
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
    handler = get_callback_handler(trace_id=trace_id, name=f"resolve:{raw_name[:40]}")
    agent   = build_resolver_agent()

    try:
        cfg = make_config(handler)
        if _USE_LANGGRAPH:
            # LangGraph agent expects HumanMessage input
            from langchain_core.messages import HumanMessage
            result = agent.invoke(
                {"messages": [HumanMessage(content=raw_name)]},
                config=cfg,
            )
            output_text = result["messages"][-1].content
        else:
            result = agent.invoke(
                {"input": raw_name},
                config=cfg,
            )
            output_text = result.get("output", "{}")

        # ── Robust JSON parsing — handles common LLM formatting issues ──
        import json, re

        def parse_llm_output(text: str) -> dict | None:
            """
            Try multiple strategies to extract a dict from LLM output.
            LLMs often return single quotes, markdown fences, Python None/True/False,
            or trailing commas — this handles all of them.
            """
            # 1. Extract content inside ```json ... ``` or ``` ... ``` fences
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if fenced:
                text = fenced.group(1)
            else:
                # 2. Extract the outermost {...} block
                braces = re.search(r"\{.*\}", text, re.DOTALL)
                if braces:
                    text = braces.group()

            # 3. Unescape Jinja/PromptTemplate double braces {{ }} → { }
            text = text.replace("{{", "{").replace("}}", "}")

            # 4. Try parsing as-is first (already valid JSON)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # 4. Fix single quotes → double quotes
            fixed = text.replace("'", '"')
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

            # 5. Fix Python literals (None, True, False → null, true, false)
            fixed = re.sub(r'\bNone\b',  'null',  fixed)
            fixed = re.sub(r'\bTrue\b',  'true',  fixed)
            fixed = re.sub(r'\bFalse\b', 'false', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

            # 6. Last resort: use ast.literal_eval (handles Python dicts natively)
            try:
                import ast
                return ast.literal_eval(text)
            except Exception:
                pass

            return None  # all strategies failed

        parsed = parse_llm_output(output_text)
        if parsed:
            return parsed

        # Could not parse anything — return raw text as evidence for debugging
        return {
            "canonical_name": None,
            "domain": None,
            "confidence": 0.0,
            "evidence": f"Parse failed. Raw output: {output_text[:300]}",
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