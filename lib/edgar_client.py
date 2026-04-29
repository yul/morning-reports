"""
SEC EDGAR ingestion client.

Provides:
  - FilingMetadata   – lightweight dataclass describing a filing
  - FilingSection    – one parsed section (name + text)
  - ParsedFiling     – all sections for a single report
  - EdgarClient      – fetches filings for tickers / latest across market
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Generator, Literal, Optional

import edgar
from edgar.company_reports import TenK, TenQ

# Only imported for type hints — avoid hard failure if XBRL data is absent
try:
    from edgar.entity.statement import FinancialStatement
except ImportError:
    FinancialStatement = None  # type: ignore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

FormType = Literal["10-K", "10-Q", "10-K/A", "10-Q/A"]
SUPPORTED_FORMS: tuple[str, ...] = ("10-K", "10-Q", "10-K/A", "10-Q/A")

# Human-readable titles for every 10-K / 10-Q item
TENK_ITEM_TITLES: dict[str, str] = {
    "Item 1":   "Business",
    "Item 1A":  "Risk Factors",
    "Item 1B":  "Unresolved Staff Comments",
    "Item 1C":  "Cybersecurity",
    "Item 2":   "Properties",
    "Item 3":   "Legal Proceedings",
    "Item 4":   "Mine Safety Disclosures",
    "Item 5":   "Market for Common Equity",
    "Item 6":   "Selected Financial Data",
    "Item 7":   "MD&A",
    "Item 7A":  "Quantitative Disclosures About Market Risk",
    "Item 8":   "Financial Statements",
    "Item 9":   "Controls and Procedures",
    "Item 9A":  "Controls and Procedures (Internal)",
    "Item 9B":  "Other Information",
    "Item 9C":  "Foreign Jurisdictions",
    "Item 10":  "Directors, Officers and Governance",
    "Item 11":  "Executive Compensation",
    "Item 12":  "Security Ownership",
    "Item 13":  "Related Transactions",
    "Item 14":  "Principal Accounting Fees",
    "Item 15":  "Exhibits and Financial Statement Schedules",
    "Item 16":  "Form 10-K Summary",
}

TENQ_ITEM_TITLES: dict[str, str] = {
    # Part I
    "Part I, Item 1":  "Financial Statements",
    "Part I, Item 2":  "MD&A",
    "Part I, Item 3":  "Quantitative Disclosures About Market Risk",
    "Part I, Item 4":  "Controls and Procedures",
    # Part II
    "Part II, Item 1": "Legal Proceedings",
    "Part II, Item 1A": "Risk Factors",
    "Part II, Item 2": "Unregistered Sales of Equity Securities",
    "Part II, Item 3": "Defaults Upon Senior Securities",
    "Part II, Item 4": "Mine Safety Disclosures",
    "Part II, Item 5": "Other Information",
    "Part II, Item 6": "Exhibits",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FilingMetadata:
    """Lightweight description of a single SEC filing (no text yet)."""
    ticker: str
    company_name: str
    cik: int
    form: str                       # "10-K" | "10-Q" | …
    accession_number: str
    filing_date: date
    period_of_report: Optional[date]
    url: str                        # EDGAR filing index URL

    @property
    def form_base(self) -> str:
        """Strip amendments: '10-K/A' → '10-K'."""
        return self.form.rstrip("/A").rstrip("/a")


@dataclass
class FilingSection:
    """One parsed section of a 10-K or 10-Q."""
    item_key: str        # e.g. "Item 7" or "Part I, Item 2"
    title: str           # human-readable title
    text: str            # plain-text content (trailing whitespace per line stripped)
    char_count: int = field(init=False)

    def __post_init__(self):
        # Strip trailing whitespace from each line — wide tables produce lots of padding
        self.text = "\n".join(line.rstrip() for line in self.text.splitlines())
        self.char_count = len(self.text)

    def is_empty(self) -> bool:
        return self.char_count < 50   # fewer than 50 chars → effectively empty


@dataclass
class FilingFinancials:
    """
    Structured financial data extracted from XBRL for one filing.

    Sourced from report.financials (not HTML text) so numbers are exact
    and multi-period comparisons are clean.

    Each statement is a pandas DataFrame (label → period columns) or None
    if the filing doesn't include that statement / XBRL is unavailable.

    markdown_*  fields are pre-rendered markdown tables ready for embedding.
    metrics     is a flat dict of the 14 most common KPIs (revenue, net_income, …)
                with None for any that couldn't be extracted.
    """
    income_statement_md: Optional[str]       # markdown table
    balance_sheet_md: Optional[str]          # markdown table
    cash_flow_md: Optional[str]              # markdown table
    metrics: dict                            # flat KPIs: revenue, net_income, …
    currency_symbol: str = "$"


@dataclass
class ParsedFiling:
    """All sections extracted from one 10-K / 10-Q filing."""
    metadata: FilingMetadata
    sections: list[FilingSection]
    financials: Optional[FilingFinancials] = None   # None if XBRL unavailable
    parse_errors: list[str] = field(default_factory=list)

    @property
    def non_empty_sections(self) -> list[FilingSection]:
        return [s for s in self.sections if not s.is_empty()]

    def get_section(self, item_key: str) -> Optional[FilingSection]:
        for s in self.sections:
            if s.item_key == item_key:
                return s
        return None


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------

class EdgarClient:
    """
    Thin wrapper around edgartools for fetching and parsing SEC filings.

    Usage
    -----
    >>> client = EdgarClient(identity="Your Name contact@yourapp.com")
    >>> meta_list = client.get_latest_filings_for_ticker("AAPL", forms=["10-K"])
    >>> parsed  = client.parse_filing(meta_list[0])
    >>> print(parsed.sections[0].text[:200])
    """

    def __init__(self, identity: Optional[str] = None):
        """
        Parameters
        ----------
        identity : str, optional
            SEC EDGAR requires a User-Agent in the format "Name email@domain.com".
            Falls back to the EDGAR_IDENTITY environment variable if not provided.
        """
        _identity = identity or os.environ.get("EDGAR_IDENTITY")
        if not _identity:
            raise ValueError(
                "EDGAR identity is required. Pass identity='Name email@domain.com' "
                "or set the EDGAR_IDENTITY environment variable."
            )
        edgar.set_identity(_identity)
        log.info("EdgarClient initialised with identity: %s", _identity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_filings_for_ticker(
        self,
        ticker: str,
        forms: list[FormType] | None = None,
        limit: int = 10,
        after: str | date | None = None,
        before: str | date | None = None,
    ) -> list[FilingMetadata]:
        """
        Return filings for a single ticker, newest first.

        Parameters
        ----------
        ticker  : stock symbol, e.g. "AAPL"
        forms   : list of form types to include; defaults to ["10-K", "10-Q"]
        limit   : max number of results
        after   : only include filings on/after this date (YYYY-MM-DD or date)
        before  : only include filings on/before this date (YYYY-MM-DD or date)
        """
        forms = forms or ["10-K", "10-Q"]
        ticker = ticker.upper()

        company = edgar.Company(ticker)
        raw_filings = company.get_filings(form=forms)

        results: list[FilingMetadata] = []
        for filing in raw_filings:
            if len(results) >= limit:
                break
            if after and filing.filing_date < _to_date(after):
                continue
            if before and filing.filing_date > _to_date(before):
                continue
            meta = self._build_metadata(ticker, filing)
            if meta:
                results.append(meta)

        log.info("Found %d filings for %s", len(results), ticker)
        return results

    def get_latest_filings(
        self,
        forms: list[FormType] | None = None,
        limit: int = 50,
    ) -> list[FilingMetadata]:
        """
        Return the most recent filings across all companies on EDGAR,
        filtered to the given form types.

        Useful for daily ingestion jobs — call this once per day to pick up
        whatever was filed since your last run.

        Parameters
        ----------
        forms : form types to include; defaults to ["10-K", "10-Q"]
        limit : max number of results to return
        """
        forms = forms or ["10-K", "10-Q"]
        current = edgar.get_current_filings()

        results: list[FilingMetadata] = []
        for filing in current:
            if len(results) >= limit:
                break
            if filing.form not in forms:
                continue
            # current_filings don't always have a ticker; try CIK-based lookup
            ticker = getattr(filing, "ticker", None) or str(filing.cik)
            meta = self._build_metadata(ticker, filing)
            if meta:
                results.append(meta)

        log.info("Found %d latest filings (forms=%s)", len(results), forms)
        return results

    def parse_filing(self, metadata: FilingMetadata) -> ParsedFiling:
        """
        Download and parse a filing into sections + structured financials.

        Returns a ParsedFiling with:
        - one FilingSection per detected item (text, trailing whitespace stripped)
        - a FilingFinancials with XBRL-sourced markdown tables and KPI metrics
          (financials is None if XBRL data is unavailable for this filing)
        """
        log.info(
            "Parsing %s %s filed %s",
            metadata.ticker, metadata.form, metadata.filing_date
        )
        raw = edgar.get_by_accession_number(metadata.accession_number)
        if raw is None:
            raise ValueError(f"Filing not found: {metadata.accession_number}")

        report = edgar.obj(raw)
        errors: list[str] = []

        if isinstance(report, TenK):
            sections = self._extract_tenk_sections(report, errors)
        elif isinstance(report, TenQ):
            sections = self._extract_tenq_sections(report, errors)
        else:
            raise ValueError(
                f"Unsupported form type: {metadata.form} "
                f"(got {type(report).__name__})"
            )

        financials = self._extract_financials(report, errors)

        if errors:
            log.warning("%d section errors for %s: %s", len(errors), metadata.accession_number, errors)

        return ParsedFiling(
            metadata=metadata,
            sections=sections,
            financials=financials,
            parse_errors=errors,
        )

    def iter_parsed_filings(
        self,
        tickers: list[str],
        forms: list[FormType] | None = None,
        limit_per_ticker: int = 10,
        after: str | date | None = None,
        before: str | date | None = None,
    ) -> Generator[ParsedFiling, None, None]:
        """
        Convenience generator: fetch + parse filings for multiple tickers.

        Yields one ParsedFiling at a time so callers can process/store
        incrementally without loading everything into memory.
        """
        for ticker in tickers:
            meta_list = self.get_filings_for_ticker(
                ticker, forms=forms, limit=limit_per_ticker,
                after=after, before=before
            )
            for meta in meta_list:
                try:
                    yield self.parse_filing(meta)
                except Exception as exc:
                    log.error("Failed to parse %s %s: %s", ticker, meta.accession_number, exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_metadata(self, ticker: str, filing) -> Optional[FilingMetadata]:
        """Convert a raw edgartools Filing into FilingMetadata."""
        try:
            return FilingMetadata(
                ticker=ticker,
                company_name=filing.company or "",
                cik=int(filing.cik),
                form=filing.form,
                accession_number=filing.accession_number,
                filing_date=filing.filing_date,
                period_of_report=_safe_date(getattr(filing.header, "period_of_report", None)),
                url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={filing.cik}"
                    f"&type={filing.form}&dateb=&owner=include&count=40",
            )
        except Exception as exc:
            log.warning("Could not build metadata for filing %s: %s", filing, exc)
            return None

    def _extract_tenk_sections(self, report: TenK, errors: list[str]) -> list[FilingSection]:
        """Extract all available items from a 10-K report."""
        sections: list[FilingSection] = []

        available_items = report.items   # e.g. ["Item 1", "Item 1A", "Item 7", ...]
        log.debug("10-K available items: %s", available_items)

        for item_key in available_items:
            title = TENK_ITEM_TITLES.get(item_key, item_key)
            try:
                text = report[item_key] or ""
                sections.append(FilingSection(item_key=item_key, title=title, text=text.strip()))
            except Exception as exc:
                errors.append(f"{item_key}: {exc}")
                log.debug("Could not extract %s: %s", item_key, exc)

        return sections

    def _extract_tenq_sections(self, report: TenQ, errors: list[str]) -> list[FilingSection]:
        """Extract all available items from a 10-Q report."""
        sections: list[FilingSection] = []

        available_items = report.items   # e.g. ["Part I, Item 1", "Part II, Item 1A", ...]
        log.debug("10-Q available items: %s", available_items)

        for item_key in available_items:
            title = TENQ_ITEM_TITLES.get(item_key, item_key)
            try:
                # TenQ.__getitem__ accepts "Part I, Item 2" style keys
                text = report[item_key] or ""
                sections.append(FilingSection(item_key=item_key, title=title, text=text.strip()))
            except Exception as exc:
                errors.append(f"{item_key}: {exc}")
                log.debug("Could not extract %s: %s", item_key, exc)

        return sections

    def _extract_financials(self, report, errors: list[str]) -> Optional[FilingFinancials]:
        """
        Extract XBRL-based financial statements as markdown tables + flat KPI metrics.

        Falls back gracefully — returns None if the filing has no XBRL data,
        and skips individual statements that fail without aborting the whole parse.

        Why markdown?  LLMs read markdown tables accurately. Numbers stay bound
        to their row labels across period columns, unlike the whitespace-aligned
        text from section.text() which can drift on unusual column widths.
        """
        try:
            fin = report.financials
            if fin is None:
                return None
        except Exception as exc:
            errors.append(f"financials: {exc}")
            return None

        def _stmt_to_markdown(getter_name: str) -> Optional[str]:
            """Call e.g. fin.income_statement() → clean markdown string, or None."""
            try:
                stmt = getattr(fin, getter_name)()
                if stmt is None:
                    return None
                df = stmt.to_numeric()
                if df is None or df.empty:
                    return None
                # Format numbers with commas; keep index (row labels)
                # pandas 2.x renamed applymap → map
                _map = getattr(df, "map", None) or df.applymap
                formatted = _map(
                    lambda v: f"{int(v):,}" if isinstance(v, float) and not __import__('math').isnan(v) else ""
                )
                return formatted.to_markdown()
            except Exception as exc:
                errors.append(f"{getter_name}: {exc}")
                return None

        def _get_metrics() -> dict:
            try:
                return fin.get_financial_metrics()
            except Exception as exc:
                errors.append(f"financial_metrics: {exc}")
                return {}

        def _get_currency() -> str:
            try:
                return fin.get_currency_symbol() or "$"
            except Exception:
                return "$"

        income_md = _stmt_to_markdown("income_statement")
        balance_md = _stmt_to_markdown("balance_sheet")
        cashflow_md = _stmt_to_markdown("cash_flow_statement")

        # Skip entirely if all three statements failed
        if income_md is None and balance_md is None and cashflow_md is None:
            log.debug("All XBRL statements empty/missing, skipping financials")
            return None

        return FilingFinancials(
            income_statement_md=income_md,
            balance_sheet_md=balance_md,
            cash_flow_md=cashflow_md,
            metrics=_get_metrics(),
            currency_symbol=_get_currency(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _safe_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        return None