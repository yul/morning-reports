"""
Supabase persistence layer.

Handles storing filings metadata, text chunks, and embeddings.
Exposes a clean interface so the ingestion pipeline only calls:

    store.upsert_filing(metadata, metrics)
    store.upsert_chunks(filing_id, embedded_chunks)
    store.search(query_embedding, ...)

Design notes
------------
- upsert everywhere — re-running ingestion on an already-stored filing
  is safe and idempotent.
- Embeddings are updated in-place on re-runs (embedded_at refreshed).
- Chunks for a filing are replaced wholesale on re-ingest to avoid
  stale chunks from old section counts surviving.
- Vector search delegates to the match_chunks() SQL function so the
  HNSW index is used by Postgres, not post-filtered in Python.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from datetime import date
from typing import Optional

from supabase import create_client, Client

from lib.edgar_client import FilingMetadata
from lib.chunker import Chunk
from lib.embedder import EmbeddedChunk

log = logging.getLogger(__name__)

# Max rows per insert batch — Supabase/PostgREST default payload limit is 1MB
_CHUNK_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class SearchResult:
    """One row returned from match_chunks()."""
    __slots__ = (
        "chunk_id", "filing_id", "accession_number", "ticker", "form",
        "filing_date", "period_of_report", "item_key", "section_title",
        "is_financial", "chunk_index", "total_chunks", "text", "similarity",
    )

    def __init__(self, row: dict):
        for k in self.__slots__:
            setattr(self, k, row.get(k))

    def __repr__(self) -> str:
        return (
            f"SearchResult(ticker={self.ticker!r}, form={self.form!r}, "
            f"section={self.section_title!r}, similarity={self.similarity:.3f})"
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class SupabaseStore:
    """
    Persistence layer backed by Supabase (Postgres + pgvector).

    Usage
    -----
    >>> store = SupabaseStore()
    >>> filing_id = store.upsert_filing(metadata, metrics={"revenue": 391035})
    >>> store.upsert_chunks(filing_id, embedded_chunks)
    >>> results = store.search(query_vec, ticker="AAPL", limit=5)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        url : Supabase project URL  (falls back to SUPABASE_URL env var)
        key : Supabase service-role key  (falls back to SUPABASE_SERVICE_KEY env var)
              Use the service-role key for backend ingestion — it bypasses RLS.
        """
        _url = url or os.environ.get("SUPABASE_URL")
        _key = key or os.environ.get("SUPABASE_SERVICE_KEY")
        if not _url or not _key:
            raise ValueError(
                "Supabase URL and service key required. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars."
            )
        self._client: Client = create_client(_url, _key)
        log.info("SupabaseStore connected to %s", _url)

    # ------------------------------------------------------------------
    # Filings
    # ------------------------------------------------------------------

    def upsert_filing(
        self,
        metadata: FilingMetadata,
        metrics: Optional[dict] = None,
    ) -> int:
        """
        Insert or update a filing row.

        Returns the filing's database id (needed to associate chunks).
        Safe to call multiple times — uses accession_number as the
        conflict key so re-ingestion doesn't create duplicates.
        """
        row = {
            "accession_number":  metadata.accession_number,
            "ticker":            metadata.ticker,
            "company_name":      metadata.company_name,
            "cik":               metadata.cik,
            "form":              metadata.form,
            "filing_date":       str(metadata.filing_date),
            "period_of_report":  str(metadata.period_of_report) if metadata.period_of_report else None,
            "url":               metadata.url,
            "financial_metrics": metrics or None,
        }

        response = (
            self._client.table("filings")
            .upsert(row, on_conflict="accession_number")
            .execute()
        )

        filing_id = response.data[0]["id"]
        log.debug("Upserted filing %s → id=%d", metadata.accession_number, filing_id)
        return filing_id

    def get_filing_id(self, accession_number: str) -> Optional[int]:
        """Look up a filing's database id by accession number."""
        response = (
            self._client.table("filings")
            .select("id")
            .eq("accession_number", accession_number)
            .maybe_single()
            .execute()
        )
        if response.data:
            return response.data["id"]
        return None

    def list_filings(
        self,
        ticker: Optional[str] = None,
        form: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return filing rows, newest first. Useful for the timeline view."""
        q = (
            self._client.table("filings")
            .select("id, accession_number, ticker, company_name, form, filing_date, period_of_report")
            .order("filing_date", desc=True)
            .limit(limit)
        )
        if ticker:
            q = q.eq("ticker", ticker.upper())
        if form:
            q = q.eq("form", form)
        return q.execute().data

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        filing_id: int,
        embedded_chunks: list[EmbeddedChunk],
    ) -> int:
        """
        Replace all chunks for a filing with the freshly-embedded set.

        Deletes existing chunks first so stale items don't accumulate
        when a filing is re-processed with different chunking settings.

        Returns the number of chunks stored.
        """
        # Delete existing chunks for this filing
        self._client.table("chunks").delete().eq("filing_id", filing_id).execute()
        log.debug("Deleted existing chunks for filing_id=%d", filing_id)

        if not embedded_chunks:
            return 0

        rows = [_embedded_to_row(filing_id, ec) for ec in embedded_chunks]
        count = 0

        for batch in _batched(rows, _CHUNK_BATCH_SIZE):
            self._client.table("chunks").insert(batch).execute()
            count += len(batch)

        log.debug("Inserted %d chunks for filing_id=%d", count, filing_id)
        return count

    def store_chunks_without_embeddings(
        self,
        filing_id: int,
        chunks: list[Chunk],
    ) -> int:
        """
        Store text chunks before embeddings are available.

        Useful when you want to inspect/validate chunking output before
        paying for embedding API calls. Embeddings can be backfilled later
        via update_embedding().
        """
        self._client.table("chunks").delete().eq("filing_id", filing_id).execute()

        rows = [_chunk_to_row(filing_id, c) for c in chunks]
        count = 0
        for batch in _batched(rows, _CHUNK_BATCH_SIZE):
            self._client.table("chunks").insert(batch).execute()
            count += len(batch)

        log.debug("Stored %d text-only chunks for filing_id=%d", count, filing_id)
        return count

    def update_embedding(self, chunk_id: int, embedding: list[float]) -> None:
        """Patch a single chunk's embedding vector."""
        from datetime import datetime, timezone
        self._client.table("chunks").update({
            "embedding": embedding,
            "embedded_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", chunk_id).execute()

    def get_unembedded_chunks(self, filing_id: Optional[int] = None) -> list[dict]:
        """Return chunks that have no embedding yet (for backfill jobs)."""
        q = (
            self._client.table("chunks")
            .select("id, filing_id, item_key, text")
            .is_("embedding", "null")
        )
        if filing_id is not None:
            q = q.eq("filing_id", filing_id)
        return q.execute().data

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        *,
        ticker: Optional[str] = None,
        form: Optional[str] = None,
        item_key: Optional[str] = None,
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Similarity search over chunk embeddings.

        Calls the match_chunks() SQL function which uses the HNSW index,
        then wraps each row in a SearchResult.

        Parameters
        ----------
        query_embedding : 512-dim vector from VoyageEmbedder.embed_query()
        ticker          : restrict to one company
        form            : "10-K" or "10-Q"
        item_key        : restrict to one section, e.g. "Item 7"
        min_date        : earliest filing_date to include
        max_date        : latest filing_date to include
        limit           : number of results
        """
        params = {
            "query_embedding": query_embedding,
            "match_count":     limit,
            "filter_ticker":   ticker.upper() if ticker else None,
            "filter_form":     form,
            "filter_item_key": item_key,
            "min_date":        str(min_date) if min_date else None,
            "max_date":        str(max_date) if max_date else None,
        }

        response = self._client.rpc("match_chunks", params).execute()
        return [SearchResult(row) for row in (response.data or [])]

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def upsert_prices(self, rows: list[dict]) -> int:
        """
        Bulk upsert OHLCV price rows.

        Each row: {"ticker": str, "date": str, "open": float,
                   "high": float, "low": float, "close": float, "volume": int}
        """
        if not rows:
            return 0
        count = 0
        for batch in _batched(rows, 500):
            self._client.table("prices").upsert(
                batch, on_conflict="ticker,date"
            ).execute()
            count += len(batch)
        log.debug("Upserted %d price rows", count)
        return count

    def get_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Return OHLCV rows for a ticker between two dates, ascending by date."""
        return (
            self._client.table("prices")
            .select("date, open, high, low, close, volume")
            .eq("ticker", ticker.upper())
            .gte("date", str(start_date))
            .lte("date", str(end_date))
            .order("date")
            .execute()
            .data
        )

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def get_score(self, filing_id: int, rule_hash: str) -> Optional[dict]:
        """Return a cached score, or None if not yet scored under this rule."""
        response = (
            self._client.table("scores")
            .select("score, rationale, scored_at")
            .eq("filing_id", filing_id)
            .eq("rule_hash", rule_hash)
            .maybe_single()
            .execute()
        )
        return response.data

    def upsert_score(
        self,
        filing_id: int,
        rule_hash: str,
        score: float,
        rationale: Optional[str] = None,
    ) -> None:
        """Store or replace a score for a (filing, rule) pair."""
        from datetime import datetime, timezone
        self._client.table("scores").upsert({
            "filing_id":  filing_id,
            "rule_hash":  rule_hash,
            "score":      score,
            "rationale":  rationale,
            "scored_at":  datetime.now(timezone.utc).isoformat(),
        }, on_conflict="filing_id,rule_hash").execute()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _chunk_to_row(filing_id: int, chunk: Chunk) -> dict:
    return {
        "filing_id":       filing_id,
        "accession_number": chunk.accession_number,
        "item_key":        chunk.item_key,
        "section_title":   chunk.section_title,
        "is_financial":    chunk.is_financial,
        "chunk_index":     chunk.chunk_index,
        "total_chunks":    chunk.total_chunks,
        "text":            chunk.text,
        "char_count":      chunk.char_count,
        "embedding":       None,
        "embedded_at":     None,
    }


def _embedded_to_row(filing_id: int, ec: EmbeddedChunk) -> dict:
    from datetime import datetime, timezone
    row = _chunk_to_row(filing_id, ec.chunk)
    row["embedding"] = ec.embedding
    row["embedded_at"] = datetime.now(timezone.utc).isoformat()
    return row


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
