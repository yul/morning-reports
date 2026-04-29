"""
Tests for store.py — no real Supabase connection required.
All Supabase client calls are mocked via a fluent builder stub.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, __import__("pathlib").Path(__file__).parent.parent.__str__())

from lib.store import SupabaseStore, SearchResult, _chunk_to_row, _embedded_to_row, _batched
from lib.edgar_client import FilingMetadata
from lib.chunker import Chunk
from lib.embedder import EmbeddedChunk


# ---------------------------------------------------------------------------
# Fluent Supabase mock builder
# ---------------------------------------------------------------------------

def make_supabase_mock(return_data=None):
    """
    Build a mock that supports the fluent Supabase pattern:
        client.table("x").select("...").eq("k","v").execute()
    Every intermediate call returns the same mock, .execute() returns
    a response object with .data = return_data.
    """
    response = MagicMock()
    response.data = return_data if return_data is not None else []

    builder = MagicMock()
    # Every chained method returns the builder itself
    for method in ("select", "insert", "upsert", "update", "delete",
                   "eq", "neq", "is_", "gte", "lte", "order", "limit",
                   "maybe_single", "single"):
        getattr(builder, method).return_value = builder
    builder.execute.return_value = response

    client = MagicMock()
    client.table.return_value = builder
    client.rpc.return_value = builder
    return client, builder, response


def make_store(client_mock) -> SupabaseStore:
    """Return a SupabaseStore with the Supabase client replaced by a mock."""
    with patch("sec_ingestion.store.create_client", return_value=client_mock):
        return SupabaseStore(url="https://fake.supabase.co", key="fake-key")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_metadata(ticker="AAPL") -> FilingMetadata:
    return FilingMetadata(
        ticker=ticker, company_name="Apple Inc.", cik=320193,
        form="10-K", accession_number="0000320193-24-000001",
        filing_date=date(2024, 11, 1), period_of_report=date(2024, 9, 28),
        url="https://sec.gov/test",
    )


def make_chunk(index=0, total=3) -> Chunk:
    return Chunk(
        ticker="AAPL", accession_number="0000320193-24-000001",
        filing_date="2024-11-01", period_of_report="2024-09-28",
        form="10-K", item_key="Item 7", section_title="MD&A",
        is_financial=False, chunk_index=index, total_chunks=total,
        text="Revenue grew 12% driven by strong iPhone sales. " * 10,
    )


def make_embedded(index=0) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=make_chunk(index=index),
        embedding=[0.01] * 512,
        model="voyage-3-lite",
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestStoreInit:
    def test_raises_without_url(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="URL"):
                SupabaseStore(key="fake")

    def test_raises_without_key(self):
        with pytest.raises(ValueError, match="key"):
            SupabaseStore(url="https://fake.supabase.co")

    def test_creates_client_with_credentials(self):
        with patch("sec_ingestion.store.create_client") as mock_cc:
            mock_cc.return_value = MagicMock()
            SupabaseStore(url="https://x.supabase.co", key="my-key")
            mock_cc.assert_called_once_with("https://x.supabase.co", "my-key")


# ---------------------------------------------------------------------------
# upsert_filing
# ---------------------------------------------------------------------------

class TestUpsertFiling:
    def test_returns_filing_id(self):
        client, builder, response = make_supabase_mock([{"id": 42}])
        store = make_store(client)
        fid = store.upsert_filing(make_metadata())
        assert fid == 42

    def test_upserts_on_accession_number(self):
        client, builder, response = make_supabase_mock([{"id": 1}])
        store = make_store(client)
        store.upsert_filing(make_metadata())
        builder.upsert.assert_called_once()
        _, kwargs = builder.upsert.call_args
        assert kwargs["on_conflict"] == "accession_number"

    def test_row_contains_all_fields(self):
        client, builder, response = make_supabase_mock([{"id": 1}])
        store = make_store(client)
        store.upsert_filing(make_metadata(), metrics={"revenue": 391035})
        row = builder.upsert.call_args[0][0]
        assert row["ticker"] == "AAPL"
        assert row["form"] == "10-K"
        assert row["cik"] == 320193
        assert row["financial_metrics"] == {"revenue": 391035}
        assert row["period_of_report"] == "2024-09-28"

    def test_period_of_report_none_when_missing(self):
        client, builder, response = make_supabase_mock([{"id": 1}])
        store = make_store(client)
        meta = make_metadata()
        meta.period_of_report = None
        store.upsert_filing(meta)
        row = builder.upsert.call_args[0][0]
        assert row["period_of_report"] is None


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------

class TestUpsertChunks:
    def test_deletes_existing_chunks_first(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        store.upsert_chunks(filing_id=1, embedded_chunks=[make_embedded()])
        # delete() should have been called with filing_id=1
        builder.delete.assert_called()
        builder.eq.assert_any_call("filing_id", 1)

    def test_inserts_correct_number_of_chunks(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        embedded = [make_embedded(i) for i in range(3)]
        count = store.upsert_chunks(filing_id=5, embedded_chunks=embedded)
        assert count == 3

    def test_embedding_stored_in_row(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        ec = make_embedded()
        store.upsert_chunks(filing_id=1, embedded_chunks=[ec])
        # Find the insert call (not the delete call)
        insert_calls = [c for c in builder.insert.call_args_list]
        assert len(insert_calls) >= 1
        rows = insert_calls[0][0][0]   # first positional arg of first insert call
        assert rows[0]["embedding"] == [0.01] * 512
        assert rows[0]["is_financial"] is False

    def test_empty_chunks_skips_insert(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        count = store.upsert_chunks(filing_id=1, embedded_chunks=[])
        assert count == 0
        builder.insert.assert_not_called()

    def test_large_batch_split_correctly(self):
        """101 chunks should generate 2 insert calls (100 + 1)."""
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        embedded = [make_embedded(i) for i in range(101)]
        store.upsert_chunks(filing_id=1, embedded_chunks=embedded)
        assert builder.insert.call_count == 2


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def _search_row(self) -> dict:
        return {
            "chunk_id": 1, "filing_id": 10, "accession_number": "x",
            "ticker": "AAPL", "form": "10-K",
            "filing_date": "2024-11-01", "period_of_report": "2024-09-28",
            "item_key": "Item 7", "section_title": "MD&A",
            "is_financial": False, "chunk_index": 0, "total_chunks": 5,
            "text": "Revenue grew 12%.", "similarity": 0.92,
        }

    def test_returns_search_results(self):
        client, builder, response = make_supabase_mock([self._search_row()])
        store = make_store(client)
        results = store.search([0.1] * 512)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].ticker == "AAPL"
        assert results[0].similarity == 0.92

    def test_passes_filters_to_rpc(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        store.search(
            [0.0] * 512,
            ticker="MSFT", form="10-Q", item_key="Item 1A",
            min_date=date(2023, 1, 1), max_date=date(2024, 12, 31),
            limit=5,
        )
        params = client.rpc.call_args[0][1]  # second positional arg
        assert params["filter_ticker"] == "MSFT"
        assert params["filter_form"] == "10-Q"
        assert params["filter_item_key"] == "Item 1A"
        assert params["min_date"] == "2023-01-01"
        assert params["max_date"] == "2024-12-31"
        assert params["match_count"] == 5

    def test_ticker_uppercased(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        store.search([0.0] * 512, ticker="aapl")
        params = client.rpc.call_args[0][1]
        assert params["filter_ticker"] == "AAPL"

    def test_no_results_returns_empty_list(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        results = store.search([0.0] * 512)
        assert results == []

    def test_calls_match_chunks_rpc(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        store.search([0.0] * 512)
        client.rpc.assert_called_once_with("match_chunks", {
            "query_embedding": [0.0] * 512,
            "match_count": 10,
            "filter_ticker": None,
            "filter_form": None,
            "filter_item_key": None,
            "min_date": None,
            "max_date": None,
        })


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

class TestScores:
    def test_get_score_returns_none_when_missing(self):
        # maybe_single() returns None in .data when no row found (real Supabase behaviour)
        client, builder, response = make_supabase_mock(None)
        response.data = None   # override: explicit None, not []
        store = make_store(client)
        result = store.get_score(filing_id=1, rule_hash="abc123")
        assert result is None

    def test_get_score_returns_data_when_present(self):
        data = {"score": 7.5, "rationale": "Strong growth.", "scored_at": "2024-11-01T00:00:00Z"}
        client, builder, response = make_supabase_mock(data)
        store = make_store(client)
        result = store.get_score(filing_id=1, rule_hash="abc123")
        assert result["score"] == 7.5

    def test_upsert_score_uses_correct_conflict_key(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        store.upsert_score(filing_id=1, rule_hash="abc", score=8.0, rationale="Good.")
        builder.upsert.assert_called_once()
        _, kwargs = builder.upsert.call_args
        assert kwargs["on_conflict"] == "filing_id,rule_hash"

    def test_upsert_score_row_values(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        store.upsert_score(filing_id=42, rule_hash="xyz", score=6.25, rationale="Moderate.")
        row = builder.upsert.call_args[0][0]
        assert row["filing_id"] == 42
        assert row["rule_hash"] == "xyz"
        assert row["score"] == 6.25
        assert row["rationale"] == "Moderate."


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

class TestPrices:
    def test_upsert_prices_batches(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        rows = [{"ticker": "AAPL", "date": f"2024-01-{i:02d}", "open": 185.0,
                 "high": 187.0, "low": 184.0, "close": 186.0, "volume": 1000000}
                for i in range(1, 12)]
        count = store.upsert_prices(rows)
        assert count == 11

    def test_upsert_prices_empty_noop(self):
        client, builder, response = make_supabase_mock([])
        store = make_store(client)
        count = store.upsert_prices([])
        assert count == 0
        builder.upsert.assert_not_called()

    def test_get_prices_filters_by_ticker_and_dates(self):
        client, builder, response = make_supabase_mock([
            {"date": "2024-01-02", "open": 185.0, "high": 187.0,
             "low": 184.0, "close": 186.5, "volume": 5000000}
        ])
        store = make_store(client)
        rows = store.get_prices("AAPL", date(2024, 1, 1), date(2024, 1, 31))
        assert len(rows) == 1
        builder.eq.assert_any_call("ticker", "AAPL")
        builder.gte.assert_any_call("date", "2024-01-01")
        builder.lte.assert_any_call("date", "2024-01-31")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_batched_even(self):
        result = list(_batched(list(range(200)), 100))
        assert len(result) == 2
        assert len(result[0]) == 100

    def test_batched_remainder(self):
        result = list(_batched(list(range(105)), 100))
        assert len(result) == 2
        assert len(result[1]) == 5

    def test_chunk_to_row_has_null_embedding(self):
        row = _chunk_to_row(7, make_chunk())
        assert row["embedding"] is None
        assert row["embedded_at"] is None
        assert row["filing_id"] == 7
        assert row["is_financial"] is False

    def test_embedded_to_row_has_vector(self):
        ec = make_embedded()
        row = _embedded_to_row(7, ec)
        assert row["embedding"] == [0.01] * 512
        assert row["embedded_at"] is not None


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_all_fields_accessible(self):
        row = {
            "chunk_id": 5, "filing_id": 10, "accession_number": "0000320193-24-000001",
            "ticker": "AAPL", "form": "10-K", "filing_date": "2024-11-01",
            "period_of_report": "2024-09-28", "item_key": "Item 7",
            "section_title": "MD&A", "is_financial": False,
            "chunk_index": 2, "total_chunks": 8,
            "text": "Revenue grew 12%.", "similarity": 0.876,
        }
        sr = SearchResult(row)
        assert sr.ticker == "AAPL"
        assert sr.similarity == 0.876
        assert sr.is_financial is False

    def test_repr_contains_key_fields(self):
        row = {"ticker": "NVDA", "form": "10-K", "section_title": "Risk Factors",
               "similarity": 0.91}
        sr = SearchResult(row)
        r = repr(sr)
        assert "NVDA" in r
        assert "0.910" in r

    def test_missing_fields_are_none(self):
        sr = SearchResult({})
        assert sr.ticker is None
        assert sr.similarity is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
