#!/usr/bin/env python3
"""
CLI for SEC filing ingestion.

Examples
--------
# List latest 10-K/Q filings across all companies
python fetch.py latest

# List latest 10-K filings only
python fetch.py latest --forms 10-K

# Fetch and parse filings for AAPL (last 5)
python fetch.py ticker AAPL --limit 5

# Fetch 10-K only, after a date
python fetch.py ticker MSFT --forms 10-K --after 2022-01-01

# Fetch multiple tickers and print all sections
python fetch.py tickers AAPL MSFT NVDA --forms 10-K --limit 2 --sections
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.edgar_client import EdgarClient, FilingMetadata, ParsedFiling

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def _require_identity() -> str:
    identity = os.environ.get("EDGAR_IDENTITY")
    if not identity:
        print(
            "Error: set EDGAR_IDENTITY env var, e.g.:\n"
            "  export EDGAR_IDENTITY='Your Name email@domain.com'",
            file=sys.stderr,
        )
        sys.exit(1)
    return identity


def print_metadata(meta: FilingMetadata):
    print(
        f"  [{meta.form:<6}] {meta.filing_date}  {meta.ticker:<6} "
        f"{meta.company_name[:45]:<45}  {meta.accession_number}"
    )


def print_parsed(parsed: ParsedFiling, show_sections: bool = False):
    m = parsed.metadata
    print(f"\n{'='*70}")
    print(f"  {m.ticker} · {m.form} · filed {m.filing_date} · period {m.period_of_report}")
    print(f"  Accession: {m.accession_number}")
    print(f"  Sections parsed: {len(parsed.non_empty_sections)} non-empty / {len(parsed.sections)} total")
    if parsed.parse_errors:
        print(f"  Errors: {parsed.parse_errors}")

    if show_sections:
        for sec in parsed.non_empty_sections:
            print(f"\n  ── {sec.item_key}: {sec.title}  ({sec.char_count:,} chars)")
            # Print first 300 chars as preview
            preview = sec.text[:300].replace("\n", " ")
            print(f"     {preview}…")


def cmd_latest(args, client: EdgarClient):
    forms = args.forms or ["10-K", "10-Q"]
    print(f"Fetching latest {args.limit} filings  forms={forms}\n")
    filings = client.get_latest_filings(forms=forms, limit=args.limit)
    for m in filings:
        print_metadata(m)


def cmd_ticker(args, client: EdgarClient):
    forms = args.forms or ["10-K", "10-Q"]
    ticker = args.ticker.upper()
    print(f"Fetching filings for {ticker}  forms={forms}  limit={args.limit}\n")
    meta_list = client.get_filings_for_ticker(
        ticker, forms=forms, limit=args.limit,
        after=args.after, before=args.before
    )
    if not meta_list:
        print("No filings found.")
        return

    if args.sections:
        for meta in meta_list:
            parsed = client.parse_filing(meta)
            print_parsed(parsed, show_sections=True)
    else:
        for meta in meta_list:
            print_metadata(meta)


def cmd_tickers(args, client: EdgarClient):
    forms = args.forms or ["10-K", "10-Q"]
    tickers = [t.upper() for t in args.tickers]
    print(f"Fetching filings for {tickers}  forms={forms}  limit/ticker={args.limit}\n")
    for parsed in client.iter_parsed_filings(
        tickers, forms=forms, limit_per_ticker=args.limit,
        after=args.after, before=args.before
    ):
        print_parsed(parsed, show_sections=args.sections)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SEC EDGAR filing fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- latest ---
    p_latest = sub.add_parser("latest", help="Latest filings across all companies")
    p_latest.add_argument("--forms", nargs="+", default=None, metavar="FORM")
    p_latest.add_argument("--limit", type=int, default=20)

    # --- ticker ---
    p_ticker = sub.add_parser("ticker", help="Filings for one ticker")
    p_ticker.add_argument("ticker")
    p_ticker.add_argument("--forms", nargs="+", default=None, metavar="FORM")
    p_ticker.add_argument("--limit", type=int, default=5)
    p_ticker.add_argument("--after", default=None, help="YYYY-MM-DD")
    p_ticker.add_argument("--before", default=None, help="YYYY-MM-DD")
    p_ticker.add_argument("--sections", action="store_true", help="Parse and show sections")

    # --- tickers ---
    p_tickers = sub.add_parser("tickers", help="Filings for multiple tickers")
    p_tickers.add_argument("tickers", nargs="+")
    p_tickers.add_argument("--forms", nargs="+", default=None, metavar="FORM")
    p_tickers.add_argument("--limit", type=int, default=3)
    p_tickers.add_argument("--after", default=None, help="YYYY-MM-DD")
    p_tickers.add_argument("--before", default=None, help="YYYY-MM-DD")
    p_tickers.add_argument("--sections", action="store_true", help="Parse and show sections")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    identity = _require_identity()
    client = EdgarClient(identity=identity)

    if args.command == "latest":
        cmd_latest(args, client)
    elif args.command == "ticker":
        cmd_ticker(args, client)
    elif args.command == "tickers":
        cmd_tickers(args, client)
