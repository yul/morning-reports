"""
Tests for chunker.py and embedder.py — no network calls required.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, __import__("pathlib").Path(__file__).parent.parent.__str__())

from lib.chunker import Chunk, SectionChunker, CHARS_PER_TOKEN
from lib.embedder import VoyageEmbedder, EmbeddedChunk, EmbeddingStats, _batch, _estimate_tokens
from lib.edgar_client import FilingMetadata, FilingSection, FilingFinancials


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_metadata(ticker="AAPL", form="10-K") -> FilingMetadata:
    return FilingMetadata(
        ticker=ticker,
        company_name="Apple Inc.",
        cik=320193,
        form=form,
        accession_number="0000320193-24-000001",
        filing_date=date(2024, 11, 1),
        period_of_report=date(2024, 9, 28),
        url="https://sec.gov/test",
    )


def make_section(item_key="Item 7", title="MD&A", text="") -> FilingSection:
    return FilingSection(item_key=item_key, title=title, text=text)


def para(n: int = 1, chars: int = 200) -> str:
    """Generate n paragraphs of `chars` characters each."""
    return "\n\n".join(("A" * chars) for _ in range(n))


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

class TestChunk:
    def _make(self, text="Hello world " * 20) -> Chunk:
        return Chunk(
            ticker="AAPL", accession_number="0000320193-24-000001",
            filing_date="2024-11-01", period_of_report="2024-09-28",
            form="10-K", item_key="Item 7", section_title="MD&A",
            is_financial=False, chunk_index=0, total_chunks=3, text=text,
        )

    def test_char_count_computed(self):
        text = "X" * 500
        c = self._make(text=text)
        assert c.char_count == 500

    def test_estimated_tokens(self):
        text = "X" * 380   # 380 / 3.8 = 100 tokens
        c = self._make(text=text)
        assert c.estimated_tokens == 100

    def test_context_prefix_contains_key_fields(self):
        c = self._make()
        prefix = c.context_prefix()
        assert "AAPL" in prefix
        assert "10-K" in prefix
        assert "MD&A" in prefix

    def test_text_for_embedding_prepends_prefix(self):
        c = self._make(text="The business grew rapidly.")
        full = c.text_for_embedding()
        assert full.startswith("Company: AAPL")
        assert "The business grew rapidly." in full


# ---------------------------------------------------------------------------
# SectionChunker — _split_text
# ---------------------------------------------------------------------------

class TestSplitText:
    def _chunker(self, max_tokens=400, overlap_tokens=50) -> SectionChunker:
        return SectionChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    def test_short_text_is_single_chunk(self):
        c = self._chunker()
        text = "Short paragraph.\n\nAnother short paragraph."
        chunks = c._split_text(text)
        assert len(chunks) == 1

    def test_long_text_splits_into_multiple(self):
        c = self._chunker(max_tokens=100)   # max_chars ≈ 380
        # 4 paragraphs of 200 chars each = 800 chars total
        text = para(n=4, chars=200)
        chunks = c._split_text(text)
        assert len(chunks) >= 2

    def test_all_content_preserved(self):
        """No text should be silently dropped."""
        c = self._chunker(max_tokens=100)
        sentences = [f"Sentence number {i} about fiscal performance." for i in range(30)]
        text = "\n\n".join(sentences)
        chunks = c._split_text(text)
        joined = " ".join(chunks)
        for s in sentences:
            assert s in joined, f"Missing: {s}"

    def test_overlap_carries_last_para(self):
        """The last paragraph of chunk N should appear at the start of chunk N+1."""
        c = self._chunker(max_tokens=80, overlap_tokens=40)
        # Each paragraph is 100 chars, overlap threshold is 40*3.8=152 chars
        paras = [f"Para {i}: " + "X" * 90 for i in range(6)]
        text = "\n\n".join(paras)
        chunks = c._split_text(text)
        assert len(chunks) >= 2
        # Last para of chunk 0 should appear in chunk 1
        last_para_of_first = chunks[0].split("\n\n")[-1]
        assert last_para_of_first in chunks[1]

    def test_empty_text_returns_empty(self):
        c = self._chunker()
        assert c._split_text("") == []
        assert c._split_text("   \n\n   ") == []

    def test_excess_newlines_normalised(self):
        c = self._chunker()
        text = "Para one.\n\n\n\n\n\nPara two."
        chunks = c._split_text(text)
        assert len(chunks) == 1
        assert "\n\n\n" not in chunks[0]

    def test_oversized_paragraph_split_by_sentences(self):
        c = self._chunker(max_tokens=50)   # max_chars ≈ 190
        # One paragraph with many sentences, total > 190 chars
        long_para = ". ".join([f"Sentence {i} about the company's performance" for i in range(10)]) + "."
        chunks = c._split_text(long_para)
        assert len(chunks) >= 2
        # No chunk exceeds 2× max_chars (table exception aside)
        for ch in chunks:
            assert len(ch) < int(50 * CHARS_PER_TOKEN) * 3


# ---------------------------------------------------------------------------
# SectionChunker — table detection
# ---------------------------------------------------------------------------

class TestTableHandling:
    def test_table_with_ruler_kept_whole(self):
        c = SectionChunker(max_tokens=50)  # tiny limit
        table = (
            "  Segment       Net Sales\n"
            " ─────────────────────────\n"
            "  Americas        167,041\n"
            "  Europe          101,328\n"
            "  China            66,952\n"
        )
        chunks = c._split_by_sentences(table)
        assert len(chunks) == 1, "Table with ruler should not be split"
        assert "167,041" in chunks[0]
        assert "66,952" in chunks[0]


# ---------------------------------------------------------------------------
# SectionChunker — chunk_filing
# ---------------------------------------------------------------------------

class TestChunkFiling:
    def _chunker(self) -> SectionChunker:
        return SectionChunker(max_tokens=100)

    def test_skips_empty_sections(self):
        c = self._chunker()
        meta = make_metadata()
        sections = [
            make_section("Item 1", "Business", "X" * 500),
            make_section("Item 1B", "Unresolved Comments", ""),   # empty
        ]
        chunks = c.chunk_filing(meta, sections)
        item_keys = {ch.item_key for ch in chunks}
        assert "Item 1B" not in item_keys

    def test_metadata_propagated_to_chunks(self):
        c = self._chunker()
        meta = make_metadata(ticker="MSFT", form="10-Q")
        sections = [make_section("Part I, Item 2", "MD&A", "X" * 500)]
        chunks = c.chunk_filing(meta, sections)
        assert all(ch.ticker == "MSFT" for ch in chunks)
        assert all(ch.form == "10-Q" for ch in chunks)
        assert all(ch.accession_number == meta.accession_number for ch in chunks)

    def test_chunk_index_sequential(self):
        c = self._chunker()
        meta = make_metadata()
        long_text = "\n\n".join(["Paragraph " + str(i) + ". " + "X" * 200 for i in range(10)])
        sections = [make_section("Item 7", "MD&A", long_text)]
        chunks = c.chunk_filing(meta, sections)
        indices = [ch.chunk_index for ch in chunks]
        assert indices == list(range(len(chunks)))

    def test_total_chunks_set_correctly(self):
        c = self._chunker()
        meta = make_metadata()
        long_text = "\n\n".join(["P" + str(i) + " " + "X" * 200 for i in range(10)])
        sections = [make_section("Item 7", "MD&A", long_text)]
        chunks = c.chunk_filing(meta, sections)
        total = len(chunks)
        assert all(ch.total_chunks == total for ch in chunks)

    def test_financial_chunks_flagged(self):
        c = self._chunker()
        meta = make_metadata()
        fin = FilingFinancials(
            income_statement_md="| Revenue | 391,035 |\n| Net Income | 93,736 |",
            balance_sheet_md=None,
            cash_flow_md=None,
            metrics={},
        )
        chunks = c.chunk_filing(meta, [], financials=fin)
        assert len(chunks) == 1
        assert chunks[0].is_financial is True
        assert "Income Statement" in chunks[0].text

    def test_all_financial_statements_included(self):
        c = self._chunker()
        meta = make_metadata()
        fin = FilingFinancials(
            income_statement_md="| Revenue | 391,035 |",
            balance_sheet_md="| Total Assets | 364,980 |",
            cash_flow_md="| Operating CF | 118,254 |",
            metrics={},
        )
        chunks = c.chunk_filing(meta, [], financials=fin)
        item_keys = {ch.item_key for ch in chunks}
        assert "financials:income_statement" in item_keys
        assert "financials:balance_sheet" in item_keys
        assert "financials:cash_flow" in item_keys

    def test_none_financials_skipped(self):
        c = self._chunker()
        meta = make_metadata()
        sections = [make_section("Item 1", "Business", "X" * 500)]
        chunks = c.chunk_filing(meta, sections, financials=None)
        assert not any(ch.is_financial for ch in chunks)


# ---------------------------------------------------------------------------
# VoyageEmbedder
# ---------------------------------------------------------------------------

class TestVoyageEmbedder:
    def _embedder(self) -> VoyageEmbedder:
        with patch("voyageai.Client"):
            return VoyageEmbedder(api_key="test-key")

    def test_raises_without_api_key(self):
        import os
        env_backup = os.environ.pop("VOYAGE_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key"):
                VoyageEmbedder()
        finally:
            if env_backup:
                os.environ["VOYAGE_API_KEY"] = env_backup

    def test_embed_query_returns_vector(self):
        embedder = self._embedder()
        fake_vector = [0.1, 0.2, 0.3] * 170 + [0.4, 0.5]  # 512-dim
        mock_response = MagicMock()
        mock_response.embeddings = [fake_vector]
        mock_response.total_tokens = 10
        embedder._client.embed.return_value = mock_response

        result = embedder.embed_query("What was Apple revenue?")
        assert isinstance(result, list)
        assert len(result) == 512
        embedder._client.embed.assert_called_once_with(
            texts=["What was Apple revenue?"],
            model=embedder.model,
            input_type="query",
            truncation=True,
        )

    def test_embed_chunks_returns_embedded_chunks(self):
        embedder = self._embedder()
        fake_vector = [0.01] * 512
        mock_response = MagicMock()
        mock_response.embeddings = [fake_vector, fake_vector]
        mock_response.total_tokens = 100
        embedder._client.embed.return_value = mock_response

        chunks = [
            Chunk(ticker="AAPL", accession_number="x", filing_date="2024-11-01",
                  period_of_report=None, form="10-K", item_key="Item 1",
                  section_title="Business", is_financial=False,
                  chunk_index=i, total_chunks=2, text=f"Text {i} " * 50)
            for i in range(2)
        ]
        results, stats = embedder.embed_chunks(chunks)
        assert len(results) == 2
        assert all(isinstance(r, EmbeddedChunk) for r in results)
        assert all(r.embedding == fake_vector for r in results)
        assert stats.total_chunks == 2
        assert stats.errors == 0

    def test_embed_chunks_uses_document_input_type(self):
        embedder = self._embedder()
        fake_vector = [0.01] * 512
        mock_response = MagicMock()
        mock_response.embeddings = [fake_vector]
        mock_response.total_tokens = 50
        embedder._client.embed.return_value = mock_response

        chunk = Chunk(ticker="AAPL", accession_number="x", filing_date="2024-11-01",
                      period_of_report=None, form="10-K", item_key="Item 7",
                      section_title="MD&A", is_financial=False,
                      chunk_index=0, total_chunks=1, text="X" * 200)
        embedder.embed_chunks([chunk])

        call_kwargs = embedder._client.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "document"

    def test_embed_chunks_batches_correctly(self):
        """200 chunks should split into 2 batches of 128 + 72."""
        embedder = self._embedder()

        def fake_embed(texts, **kwargs):
            resp = MagicMock()
            resp.embeddings = [[0.0] * 512 for _ in texts]
            resp.total_tokens = len(texts) * 100
            return resp

        embedder._client.embed.side_effect = fake_embed

        chunks = [
            Chunk(ticker="AAPL", accession_number="x", filing_date="2024-01-01",
                  period_of_report=None, form="10-K", item_key="Item 1",
                  section_title="Business", is_financial=False,
                  chunk_index=i, total_chunks=200, text=f"chunk {i} " * 30)
            for i in range(200)
        ]
        results, stats = embedder.embed_chunks(chunks)
        assert len(results) == 200
        assert stats.batches == 2   # 128 + 72
        assert embedder._client.embed.call_count == 2

    def test_retries_on_rate_limit(self):
        embedder = self._embedder()
        fake_vector = [0.0] * 512
        good_response = MagicMock()
        good_response.embeddings = [fake_vector]
        good_response.total_tokens = 50

        # Fail once with 429, then succeed
        embedder._client.embed.side_effect = [
            Exception("429 Too Many Requests"),
            good_response,
        ]

        chunk = Chunk(ticker="AAPL", accession_number="x", filing_date="2024-01-01",
                      period_of_report=None, form="10-K", item_key="Item 1",
                      section_title="Business", is_financial=False,
                      chunk_index=0, total_chunks=1, text="X" * 100)

        with patch("time.sleep"):   # don't actually wait
            results, stats = embedder.embed_chunks([chunk])

        assert len(results) == 1
        assert stats.errors == 0
        assert embedder._client.embed.call_count == 2

    def test_gives_up_after_max_retries(self):
        embedder = self._embedder()
        embedder._client.embed.side_effect = Exception("500 Server Error")

        chunk = Chunk(ticker="AAPL", accession_number="x", filing_date="2024-01-01",
                      period_of_report=None, form="10-K", item_key="Item 1",
                      section_title="Business", is_financial=False,
                      chunk_index=0, total_chunks=1, text="X" * 100)

        with patch("time.sleep"):
            results, stats = embedder.embed_chunks([chunk])

        assert len(results) == 0
        assert stats.errors == 1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_batch_even_split(self):
        items = list(range(256))
        batches = _batch(items, 128)
        assert len(batches) == 2
        assert batches[0] == list(range(128))
        assert batches[1] == list(range(128, 256))

    def test_batch_remainder(self):
        items = list(range(130))
        batches = _batch(items, 128)
        assert len(batches) == 2
        assert len(batches[1]) == 2

    def test_batch_smaller_than_size(self):
        batches = _batch(list(range(10)), 128)
        assert len(batches) == 1

    def test_estimate_tokens(self):
        texts = ["A" * 380, "B" * 380]   # 760 chars → ~200 tokens
        tokens = _estimate_tokens(texts)
        assert 190 <= tokens <= 210


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
