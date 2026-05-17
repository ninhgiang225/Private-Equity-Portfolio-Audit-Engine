"""src/agent/tools.py

Three tools the ReAct agent can call:
  1. fuzzy_match_tool    — fast local string similarity (no API needed)
  2. web_search_tool     — Tavily web search for unknown companies
  3. company_lookup_tool — mock database of known canonical names

Start with fuzzy_match and company_lookup (free, no API key).
Add web_search once you have a Tavily key.
"""
from langchain_core.tools import tool
from rapidfuzz import process, fuzz
from typing import Optional
import json

# ── Canonical name database (mock — replace with real DB / BigQuery later) ──
KNOWN_COMPANIES: dict[str, dict] = {
    "apple inc.":           {"canonical": "Apple Inc.",           "domain": "apple.com"},
    "microsoft corporation":{"canonical": "Microsoft Corporation","domain": "microsoft.com"},
    "alphabet inc.":        {"canonical": "Alphabet Inc.",        "domain": "abc.xyz"},
    "amazon.com inc.":      {"canonical": "Amazon.com Inc.",      "domain": "amazon.com"},
    "meta platforms inc.":  {"canonical": "Meta Platforms Inc.",  "domain": "meta.com"},
    "tesla inc.":           {"canonical": "Tesla Inc.",           "domain": "tesla.com"},
    "nvidia corporation":   {"canonical": "NVIDIA Corporation",   "domain": "nvidia.com"},
    "salesforce inc.":      {"canonical": "Salesforce Inc.",      "domain": "salesforce.com"},
    "adobe inc.":           {"canonical": "Adobe Inc.",           "domain": "adobe.com"},
    "oracle corporation":   {"canonical": "Oracle Corporation",   "domain": "oracle.com"},
    "servicenow inc.":      {"canonical": "ServiceNow Inc.",      "domain": "servicenow.com"},
    "workday inc.":         {"canonical": "Workday Inc.",         "domain": "workday.com"},
    "shopify inc.":         {"canonical": "Shopify Inc.",         "domain": "shopify.com"},
    "snowflake inc.":       {"canonical": "Snowflake Inc.",       "domain": "snowflake.com"},
    "datadog inc.":         {"canonical": "Datadog Inc.",         "domain": "datadoghq.com"},
    "mongodb inc.":         {"canonical": "MongoDB Inc.",         "domain": "mongodb.com"},
    "confluent inc.":       {"canonical": "Confluent Inc.",       "domain": "confluent.io"},
    "hashicorp inc.":       {"canonical": "HashiCorp Inc.",       "domain": "hashicorp.com"},
    "twilio inc.":          {"canonical": "Twilio Inc.",          "domain": "twilio.com"},
    "zendesk inc.":         {"canonical": "Zendesk Inc.",         "domain": "zendesk.com"},
    "hubspot inc.":         {"canonical": "HubSpot Inc.",         "domain": "hubspot.com"},
    "asana inc.":           {"canonical": "Asana Inc.",           "domain": "asana.com"},
    "notion labs inc.":     {"canonical": "Notion Labs Inc.",     "domain": "notion.so"},
    "figma inc.":           {"canonical": "Figma Inc.",           "domain": "figma.com"},
    "stripe inc.":          {"canonical": "Stripe Inc.",          "domain": "stripe.com"},
    "plaid inc.":           {"canonical": "Plaid Inc.",           "domain": "plaid.com"},
    "brex inc.":            {"canonical": "Brex Inc.",            "domain": "brex.com"},
    "rippling inc.":        {"canonical": "Rippling Inc.",        "domain": "rippling.com"},
    "lattice inc.":         {"canonical": "Lattice Inc.",         "domain": "lattice.com"},
    "scale ai inc.":        {"canonical": "Scale AI Inc.",        "domain": "scale.com"},
    "weights & biases inc.":{"canonical": "Weights & Biases Inc.","domain": "wandb.ai"},
}


@tool
def fuzzy_match_tool(company_name: str, threshold: int = 75) -> str:
    """
    Find the closest canonical company name using fuzzy string matching.
    Returns the best match and its similarity score (0-100).
    Use this as a first-pass before trying web search.

    Args:
        company_name: The raw/messy company name to match.
        threshold: Minimum similarity score to return a result (default 75).
    """
    query = company_name.lower().strip()
    candidates = list(KNOWN_COMPANIES.keys())

    # Try token sort ratio (handles word-order differences well)
    result = process.extractOne(
        query,
        candidates,
        scorer=fuzz.token_sort_ratio,
    )

    if result is None or result[1] < threshold:
        return json.dumps({
            "match_found": False,
            "message": f"No fuzzy match above threshold {threshold} for '{company_name}'",
        })

    matched_key, score, _ = result
    entry = KNOWN_COMPANIES[matched_key]
    return json.dumps({
        "match_found": True,
        "canonical_name": entry["canonical"],
        "domain": entry["domain"],
        "similarity_score": score,
        "evidence": f"Fuzzy matched '{company_name}' → '{entry['canonical']}' (score {score}/100)",
    })


@tool
def company_lookup_tool(company_name: str) -> str:
    """
    Perform an exact (case-insensitive) lookup of a company name in the
    canonical database. Use this before fuzzy_match for clean inputs.

    Args:
        company_name: The company name to look up.
    """
    key = company_name.lower().strip()

    if key in KNOWN_COMPANIES:
        entry = KNOWN_COMPANIES[key]
        return json.dumps({
            "match_found": True,
            "canonical_name": entry["canonical"],
            "domain": entry["domain"],
            "evidence": f"Exact match found in canonical database.",
        })

    return json.dumps({
        "match_found": False,
        "message": f"'{company_name}' not found in canonical database. Try fuzzy_match or web_search.",
    })


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web to identify a company and find its canonical name.
    Use this for companies not found by fuzzy_match or company_lookup.
    Requires TAVILY_API_KEY to be set.

    Args:
        query: A search query like 'What company is TSLA? canonical full name'.
    """
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        from src.utils.config import TAVILY_API_KEY
        import os
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

        search = TavilySearchResults(max_results=3)
        results = search.invoke(query)

        if not results:
            return json.dumps({"match_found": False, "message": "No web results found."})

        # Return top result as evidence for the agent to reason about
        top = results[0]
        return json.dumps({
            "match_found": True,
            "evidence": top.get("content", "")[:500],  # truncate for token efficiency
            "source_url": top.get("url", ""),
        })

    except ImportError:
        return json.dumps({
            "match_found": False,
            "message": "Tavily not installed or TAVILY_API_KEY not set. pip install tavily-python",
        })
    except Exception as e:
        return json.dumps({"match_found": False, "message": str(e)})


# ── All tools in one list (imported by resolver.py) ──────────────────────────
ALL_TOOLS = [company_lookup_tool, fuzzy_match_tool, web_search_tool]
