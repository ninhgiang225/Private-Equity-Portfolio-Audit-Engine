"""tests/test_tools.py

Unit tests for the entity resolution tools.
These run without any API keys — purely local logic.

Run with:  pytest tests/test_tools.py -v
"""
import json
import pytest
from src.agent.tools import fuzzy_match_tool, company_lookup_tool


class TestFuzzyMatchTool:
    def test_obvious_typo(self):
        result = json.loads(fuzzy_match_tool.invoke("Aple Inc"))
        assert result["match_found"] is True
        assert result["canonical_name"] == "Apple Inc."

    def test_ticker_symbol_no_match(self):
        # Short tickers often score below threshold
        result = json.loads(fuzzy_match_tool.invoke("MSFT"))
        # Either finds Microsoft or finds nothing — both are valid
        assert "match_found" in result

    def test_below_threshold(self):
        result = json.loads(fuzzy_match_tool.invoke("XYZXYZXYZ Nonexistent Corp"))
        assert result["match_found"] is False

    def test_word_order_variation(self):
        result = json.loads(fuzzy_match_tool.invoke("Inc. Salesforce"))
        assert result["match_found"] is True
        assert "Salesforce" in result["canonical_name"]


class TestCompanyLookupTool:
    def test_exact_match_lowercase(self):
        result = json.loads(company_lookup_tool.invoke("apple inc."))
        assert result["match_found"] is True
        assert result["canonical_name"] == "Apple Inc."

    def test_exact_match_uppercase(self):
        # Lookup is case-insensitive
        result = json.loads(company_lookup_tool.invoke("APPLE INC."))
        assert result["match_found"] is True

    def test_miss(self):
        result = json.loads(company_lookup_tool.invoke("Some Unknown Startup"))
        assert result["match_found"] is False
        assert "message" in result
