"""
Unit tests for edgar_client.py — no network calls required.

These tests mock edgartools objects to validate parsing logic,
metadata construction, and section extraction independently of SEC access.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.edgar_client import (
    EdgarClient,
    FilingMetadata,
    FilingSection,
    FilingFinancials,
    ParsedFiling,
    TENK_ITEM_TITLES,
    TENQ_ITEM_TITLES,
    _to_date,
    _safe_date,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_raw_filing(
    ticker="AAPL",
    company="Apple Inc.",
    cik=320193,
    form="10-K",
    accession="0000320193-24-000001",
    filing_date=date(2024, 11, 1),
    period="2024-09-28",
):
    f = MagicMock()
    f.ticker = ticker
    f.company = company
    f.cik = cik
    f.form = form
    f.accession_number = accession
    f.filing_date = filing_date
    f.header.period_of_report = period
    return f


def make_mock_tenk(items: list[str], sections: dict[str, str]) -> MagicMock:
    """Create a mock TenK report."""
    from edgar.company_reports import TenK
    report = MagicMock(spec=TenK)
    report.items = items
    report.__getitem__ = lambda self, key: sections.get(key, "")
    return report


def make_mock_tenq(items: list[str], sections: dict[str, str]) -> MagicMock:
    """Create a mock TenQ report."""
    from edgar.company_reports import TenQ
    report = MagicMock(spec=TenQ)
    report.items = items
    report.__getitem__ = lambda self, key: sections.get(key, "")
    return report


# ---------------------------------------------------------------------------
# FilingMetadata
# ---------------------------------------------------------------------------

class TestFilingMetadata:
    def test_form_base_strips_amendment(self):
        m = FilingMetadata(
            ticker="AAPL", company_name="Apple", cik=1,
            form="10-K/A", accession_number="x", filing_date=date.today(),
            period_of_report=None, url="https://sec.gov"
        )
        assert m.form_base == "10-K"

    def test_form_base_unchanged_for_standard(self):
        m = FilingMetadata(
            ticker="MSFT", company_name="Microsoft", cik=2,
            form="10-Q", accession_number="x", filing_date=date.today(),
            period_of_report=None, url="https://sec.gov"
        )
        assert m.form_base == "10-Q"


# ---------------------------------------------------------------------------
# FilingSection
# ---------------------------------------------------------------------------

class TestFilingSection:
    def test_char_count_computed(self):
        s = FilingSection(item_key="Item 1", title="Business", text="Hello world")
        assert s.char_count == len("Hello world")

    def test_is_empty_short_text(self):
        s = FilingSection(item_key="Item 1B", title="X", text="Short")
        assert s.is_empty() is True

    def test_is_empty_long_text(self):
        s = FilingSection(item_key="Item 7", title="MD&A", text="A" * 100)
        assert s.is_empty() is False


# ---------------------------------------------------------------------------
# ParsedFiling
# ---------------------------------------------------------------------------

class TestParsedFiling:
    def _make_parsed(self, texts: list[str], financials=None) -> ParsedFiling:
        meta = FilingMetadata(
            ticker="TEST", company_name="Test Co", cik=99,
            form="10-K", accession_number="0", filing_date=date.today(),
            period_of_report=None, url=""
        )
        sections = [
            FilingSection(item_key=f"Item {i}", title=f"Section {i}", text=t)
            for i, t in enumerate(texts, 1)
        ]
        return ParsedFiling(metadata=meta, sections=sections, financials=financials)

    def test_non_empty_sections_filters_short(self):
        parsed = self._make_parsed(["A" * 200, "tiny", "B" * 300])
        assert len(parsed.non_empty_sections) == 2

    def test_get_section_by_key(self):
        parsed = self._make_parsed(["A" * 200])
        assert parsed.get_section("Item 1") is not None
        assert parsed.get_section("Item 99") is None

    def test_financials_none_by_default(self):
        parsed = self._make_parsed(["A" * 200])
        assert parsed.financials is None

    def test_financials_attached(self):
        fin = FilingFinancials(
            income_statement_md="| Net Income | 1000 |",
            balance_sheet_md=None,
            cash_flow_md=None,
            metrics={"revenue": 5000, "net_income": 1000},
        )
        parsed = self._make_parsed(["A" * 200], financials=fin)
        assert parsed.financials is not None
        assert parsed.financials.metrics["revenue"] == 5000


# ---------------------------------------------------------------------------
# FilingSection whitespace stripping
# ---------------------------------------------------------------------------

class TestFilingSectionWhitespace:
    def test_trailing_whitespace_stripped_per_line(self):
        # Simulate wide table padding that edgartools produces
        raw = "  Products     $294,866     $298,085     \n  Services       96,169       85,200     "
        sec = FilingSection(item_key="Item 8", title="Financial Statements", text=raw)
        for line in sec.text.splitlines():
            assert not line.endswith(" "), f"trailing space on line: {repr(line)}"

    def test_content_preserved_after_strip(self):
        raw = "  Revenue   $391,035   $383,285   \n  Net Income  $93,736   $96,995   "
        sec = FilingSection(item_key="Item 7", title="MD&A", text=raw)
        assert "$391,035" in sec.text
        assert "$93,736" in sec.text

    def test_char_count_reflects_stripped_text(self):
        padded = "Hello      "
        sec = FilingSection(item_key="Item 1", title="Business", text=padded)
        assert sec.char_count == len("Hello")  # trailing spaces removed


# ---------------------------------------------------------------------------
# FilingFinancials
# ---------------------------------------------------------------------------

class TestFilingFinancials:
    def test_defaults(self):
        fin = FilingFinancials(
            income_statement_md="| a | b |",
            balance_sheet_md=None,
            cash_flow_md=None,
            metrics={},
        )
        assert fin.currency_symbol == "$"

    def test_all_fields_accessible(self):
        fin = FilingFinancials(
            income_statement_md="| Net Income | 93,736 |",
            balance_sheet_md="| Total Assets | 364,980 |",
            cash_flow_md="| Operating CF | 118,254 |",
            metrics={
                "revenue": 391035,
                "net_income": 93736,
                "operating_cash_flow": 118254,
                "free_cash_flow": 87018,
            },
            currency_symbol="$",
        )
        assert "Net Income" in fin.income_statement_md
        assert "Total Assets" in fin.balance_sheet_md
        assert "Operating CF" in fin.cash_flow_md
        assert fin.metrics["free_cash_flow"] == 87018


# ---------------------------------------------------------------------------
# EdgarClient._extract_financials
# ---------------------------------------------------------------------------

class TestExtractFinancials:
    def _client(self) -> EdgarClient:
        with patch("edgar.set_identity"):
            return EdgarClient(identity="Test test@test.com")

    def _make_stmt_mock(self, data: dict) -> MagicMock:
        """Make a mock FinancialStatement that returns a real DataFrame."""
        import pandas as pd
        df = pd.DataFrame(data)
        stmt = MagicMock()
        stmt.to_numeric.return_value = df
        return stmt

    def test_returns_none_when_financials_is_none(self):
        client = self._client()
        report = MagicMock()
        report.financials = None
        errors = []
        result = client._extract_financials(report, errors)
        assert result is None
        assert errors == []

    def test_returns_none_when_financials_raises(self):
        client = self._client()
        report = MagicMock()
        type(report).financials = property(lambda self: (_ for _ in ()).throw(RuntimeError("no xbrl")))
        errors = []
        result = client._extract_financials(report, errors)
        assert result is None
        assert any("financials" in e for e in errors)

    def test_returns_none_when_all_statements_empty(self):
        client = self._client()
        fin = MagicMock()
        fin.income_statement.return_value = None
        fin.balance_sheet.return_value = None
        fin.cash_flow_statement.return_value = None
        fin.get_financial_metrics.return_value = {}
        fin.get_currency_symbol.return_value = "$"
        report = MagicMock()
        report.financials = fin
        errors = []
        result = client._extract_financials(report, errors)
        assert result is None

    def test_produces_markdown_when_statements_available(self):
        import pandas as pd
        client = self._client()

        df = pd.DataFrame(
            {"2024": [391035.0, 93736.0], "2023": [383285.0, 96995.0]},
            index=["Total Revenue", "Net Income"]
        )
        stmt_mock = MagicMock()
        stmt_mock.to_numeric.return_value = df

        fin = MagicMock()
        fin.income_statement.return_value = stmt_mock
        fin.balance_sheet.return_value = None
        fin.cash_flow_statement.return_value = None
        fin.get_financial_metrics.return_value = {"revenue": 391035, "net_income": 93736}
        fin.get_currency_symbol.return_value = "$"

        report = MagicMock()
        report.financials = fin
        errors = []
        result = client._extract_financials(report, errors)

        assert result is not None
        assert result.income_statement_md is not None
        assert "Total Revenue" in result.income_statement_md
        assert "391,035" in result.income_statement_md
        assert result.balance_sheet_md is None
        assert result.metrics["revenue"] == 391035
        assert result.currency_symbol == "$"

    def test_individual_statement_error_recorded_but_others_continue(self):
        import pandas as pd
        client = self._client()

        df = pd.DataFrame({"2024": [100000.0]}, index=["Total Assets"])
        stmt_mock = MagicMock()
        stmt_mock.to_numeric.return_value = df

        fin = MagicMock()
        fin.income_statement.side_effect = RuntimeError("xbrl missing")
        fin.balance_sheet.return_value = stmt_mock
        fin.cash_flow_statement.return_value = None
        fin.get_financial_metrics.return_value = {}
        fin.get_currency_symbol.return_value = "$"

        report = MagicMock()
        report.financials = fin
        errors = []
        result = client._extract_financials(report, errors)

        assert result is not None
        assert result.income_statement_md is None   # failed
        assert result.balance_sheet_md is not None  # succeeded
        assert any("income_statement" in e for e in errors)


# ---------------------------------------------------------------------------
# EdgarClient._build_metadata
# ---------------------------------------------------------------------------

class TestBuildMetadata:
    def _client(self) -> EdgarClient:
        with patch("edgar.set_identity"):
            return EdgarClient(identity="Test test@test.com")

    def test_builds_metadata_correctly(self):
        client = self._client()
        raw = make_mock_raw_filing()
        meta = client._build_metadata("AAPL", raw)
        assert meta is not None
        assert meta.ticker == "AAPL"
        assert meta.form == "10-K"
        assert meta.cik == 320193
        assert meta.filing_date == date(2024, 11, 1)

    def test_returns_none_on_exception(self):
        client = self._client()
        bad = MagicMock()
        # header.period_of_report raises an exception → _build_metadata catches it
        bad.cik = 12345
        bad.company = "Test Co"
        bad.form = "10-K"
        bad.accession_number = "0000012345-24-000001"
        bad.filing_date = date(2024, 1, 1)
        type(bad).header = property(lambda self: (_ for _ in ()).throw(RuntimeError("broken header")))
        result = client._build_metadata("XX", bad)
        assert result is None


# ---------------------------------------------------------------------------
# EdgarClient._extract_tenk_sections
# ---------------------------------------------------------------------------

class TestExtractTenkSections:
    def _client(self) -> EdgarClient:
        with patch("edgar.set_identity"):
            return EdgarClient(identity="Test test@test.com")

    def test_extracts_all_items(self):
        client = self._client()
        report = make_mock_tenk(
            items=["Item 1", "Item 7", "Item 1A"],
            sections={
                "Item 1": "Business content " * 20,
                "Item 7": "MD&A content " * 20,
                "Item 1A": "Risk factors " * 20,
            }
        )
        errors = []
        sections = client._extract_tenk_sections(report, errors)
        assert len(sections) == 3
        assert errors == []

    def test_uses_correct_titles(self):
        client = self._client()
        report = make_mock_tenk(
            items=["Item 7"],
            sections={"Item 7": "MD&A text " * 20},
        )
        sections = client._extract_tenk_sections(report, [])
        assert sections[0].title == "MD&A"

    def test_unknown_item_uses_key_as_title(self):
        client = self._client()
        report = make_mock_tenk(
            items=["Item 99"],
            sections={"Item 99": "Unknown section " * 20},
        )
        sections = client._extract_tenk_sections(report, [])
        assert sections[0].title == "Item 99"

    def test_records_errors_on_exception(self):
        client = self._client()
        report = MagicMock()
        report.items = ["Item 7"]
        report.__getitem__ = MagicMock(side_effect=RuntimeError("parse failed"))
        errors = []
        sections = client._extract_tenk_sections(report, errors)
        assert len(sections) == 0
        assert len(errors) == 1
        assert "Item 7" in errors[0]

    def test_empty_text_included_but_flagged(self):
        client = self._client()
        report = make_mock_tenk(
            items=["Item 1B"],
            sections={"Item 1B": ""},   # empty
        )
        sections = client._extract_tenk_sections(report, [])
        assert len(sections) == 1
        assert sections[0].is_empty()


# ---------------------------------------------------------------------------
# EdgarClient._extract_tenq_sections
# ---------------------------------------------------------------------------

class TestExtractTenqSections:
    def _client(self) -> EdgarClient:
        with patch("edgar.set_identity"):
            return EdgarClient(identity="Test test@test.com")

    def test_extracts_part_qualified_items(self):
        client = self._client()
        report = make_mock_tenq(
            items=["Part I, Item 1", "Part I, Item 2", "Part II, Item 1A"],
            sections={
                "Part I, Item 1": "Financial statements " * 20,
                "Part I, Item 2": "MDA content " * 20,
                "Part II, Item 1A": "Risk factors " * 20,
            }
        )
        errors = []
        sections = client._extract_tenq_sections(report, errors)
        assert len(sections) == 3
        assert sections[0].item_key == "Part I, Item 1"
        assert sections[0].title == "Financial Statements"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_to_date_from_string(self):
        assert _to_date("2024-01-15") == date(2024, 1, 15)

    def test_to_date_from_date(self):
        d = date(2023, 6, 1)
        assert _to_date(d) is d

    def test_safe_date_none(self):
        assert _safe_date(None) is None

    def test_safe_date_string(self):
        assert _safe_date("2022-03-31") == date(2022, 3, 31)

    def test_safe_date_invalid_returns_none(self):
        assert _safe_date("not-a-date") is None


# ---------------------------------------------------------------------------
# Item title coverage
# ---------------------------------------------------------------------------

class TestItemTitles:
    def test_tenk_covers_all_standard_items(self):
        standard = [f"Item {x}" for x in
                    ["1", "1A", "1B", "1C", "2", "3", "4",
                     "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
                     "10", "11", "12", "13", "14", "15", "16"]]
        missing = [k for k in standard if k not in TENK_ITEM_TITLES]
        assert missing == [], f"Missing 10-K item titles: {missing}"

    def test_tenq_covers_all_standard_items(self):
        standard = [
            "Part I, Item 1", "Part I, Item 2", "Part I, Item 3", "Part I, Item 4",
            "Part II, Item 1", "Part II, Item 1A", "Part II, Item 2",
            "Part II, Item 3", "Part II, Item 4", "Part II, Item 5", "Part II, Item 6",
        ]
        missing = [k for k in standard if k not in TENQ_ITEM_TITLES]
        assert missing == [], f"Missing 10-Q item titles: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])