"""
Text chunker for SEC filing sections.

Splits FilingSection text into overlapping chunks at natural paragraph
boundaries, keeping tables intact. Produces Chunk objects that carry
enough metadata to reconstruct context (ticker, form, section, position).

Design decisions:
  - Split at double-newline (paragraph) boundaries first; fall back to
    sentence boundaries only when a single paragraph exceeds max_chars.
  - Tables (detected by ─── ruler lines or consistent spacing patterns)
    are kept whole wherever possible — never split mid-row.
  - Overlap is one paragraph (the last paragraph of the previous chunk),
    giving the embedding model sentence-level context continuity.
  - Chunk size targets ~400 tokens (≈ 1,500 chars for mixed SEC text).
    Hard ceiling at ~500 tokens (≈ 1,900 chars) to stay inside
    voyage-3-lite's optimal range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from sec_ingestion.edgar_client import FilingMetadata, FilingSection, FilingFinancials


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters per token estimate for mixed SEC prose + tables.
# Prose ≈ 4 chars/token; number-heavy tables ≈ 3 chars/token.
# We use 3.8 as a conservative average.
CHARS_PER_TOKEN: float = 3.8

DEFAULT_MAX_TOKENS: int = 400      # target chunk ceiling
DEFAULT_OVERLAP_TOKENS: int = 50   # one short paragraph of overlap

# Regex that matches a horizontal rule drawn by edgartools table renderer
_TABLE_RULE_RE = re.compile(r"^[\s─\-─]+$")
# Matches a line that looks like a table data row (2+ runs of spaces between content)
_TABLE_ROW_RE = re.compile(r"\S+\s{2,}\S")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    One embeddable unit of text from a filing section.

    Fields prefixed with filing_ / section_ carry the metadata needed to
    store the chunk in Supabase and reconstruct the retrieval context.
    """
    # Filing identity
    ticker: str
    accession_number: str
    filing_date: str             # ISO date string  "2024-11-01"
    period_of_report: Optional[str]  # ISO date string or None
    form: str                    # "10-K" | "10-Q"

    # Section identity
    item_key: str                # "Item 7"  |  "Part I, Item 2"
    section_title: str           # "MD&A"
    is_financial: bool           # True → came from XBRL markdown table

    # Chunk position within the section
    chunk_index: int             # 0-based position within this section
    total_chunks: int            # total chunks in this section (set after splitting)

    # Content
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    @property
    def estimated_tokens(self) -> int:
        return int(self.char_count / CHARS_PER_TOKEN)

    def context_prefix(self) -> str:
        """
        Prepend this to the chunk text before embedding for better retrieval.
        Voyage AI recommends prepending document context for 'document' input_type.
        """
        return (
            f"Company: {self.ticker} | Form: {self.form} | "
            f"Period: {self.period_of_report or self.filing_date} | "
            f"Section: {self.section_title}"
        )

    def text_for_embedding(self) -> str:
        """Full text sent to the embedding model (prefix + content)."""
        return f"{self.context_prefix()}\n\n{self.text}"


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class SectionChunker:
    """
    Splits a filing's sections into Chunk objects.

    Usage
    -----
    >>> chunker = SectionChunker()
    >>> chunks = chunker.chunk_filing(metadata, sections, financials)
    >>> print(len(chunks), "chunks total")
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ):
        self.max_chars = int(max_tokens * CHARS_PER_TOKEN)
        self.overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_filing(
        self,
        metadata: FilingMetadata,
        sections: list[FilingSection],
        financials: Optional[FilingFinancials] = None,
    ) -> list[Chunk]:
        """
        Produce all chunks for a filing.

        Text sections and XBRL financial tables are chunked separately:
        - Text sections  → paragraph-based overlap chunking
        - Financial MDs  → each statement is one chunk (already compact markdown)
        """
        chunks: list[Chunk] = []

        for section in sections:
            if section.is_empty():
                continue
            section_chunks = self._chunk_section(metadata, section)
            chunks.extend(section_chunks)

        if financials is not None:
            fin_chunks = self._chunk_financials(metadata, financials)
            chunks.extend(fin_chunks)

        return chunks

    # ------------------------------------------------------------------
    # Text section chunking
    # ------------------------------------------------------------------

    def _chunk_section(
        self, metadata: FilingMetadata, section: FilingSection
    ) -> list[Chunk]:
        """Split one text section into overlapping chunks."""
        raw_chunks = self._split_text(section.text)

        result: list[Chunk] = []
        total = len(raw_chunks)
        for i, text in enumerate(raw_chunks):
            result.append(self._make_chunk(
                metadata=metadata,
                item_key=section.item_key,
                section_title=section.title,
                is_financial=False,
                chunk_index=i,
                total_chunks=total,
                text=text,
            ))
        return result

    def _split_text(self, text: str) -> list[str]:
        """
        Core splitting logic.

        1. Normalise whitespace (collapse 3+ newlines → 2).
        2. Split on double-newlines into paragraphs.
        3. Accumulate paragraphs into chunks ≤ max_chars.
        4. If a single paragraph exceeds max_chars, fall back to sentence splitting.
        5. Overlap: carry the last paragraph of each chunk into the next.
        """
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_len: int = 0

        for para in paragraphs:
            para_len = len(para)

            if para_len > self.max_chars:
                # Flush what we have first
                if current:
                    chunks.append("\n\n".join(current))
                    current, current_len = [], 0
                # Then split the oversized paragraph by sentences
                for sub in self._split_by_sentences(para):
                    chunks.append(sub)
                continue

            sep = 2  # the \n\n separator
            if current_len + para_len + sep > self.max_chars and current:
                chunks.append("\n\n".join(current))
                # Overlap: keep last paragraph if it's short enough
                if current and len(current[-1]) <= self.overlap_chars:
                    current = [current[-1], para]
                    current_len = len(current[-1]) + para_len + sep
                else:
                    current = [para]
                    current_len = para_len
            else:
                current.append(para)
                current_len += para_len + sep

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        """
        Fallback for paragraphs that exceed max_chars (e.g. run-on risk factors).

        Splits on sentence boundaries. Tables detected by the ruler pattern
        are kept whole: the entire table block is emitted as one chunk even
        if it exceeds max_chars (better than splitting mid-row).
        """
        # Detect table blocks (ruler line present) — keep whole
        if _TABLE_RULE_RE.search(text):
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len + 1 > self.max_chars and current:
                chunks.append(" ".join(current))
                overlap = [current[-1]] if len(current[-1]) <= self.overlap_chars else []
                current = overlap + [sent]
                current_len = sum(len(s) for s in current)
            else:
                current.append(sent)
                current_len += sent_len + 1

        if current:
            chunks.append(" ".join(current))

        return chunks

    # ------------------------------------------------------------------
    # Financial markdown chunking
    # ------------------------------------------------------------------

    def _chunk_financials(
        self, metadata: FilingMetadata, financials: FilingFinancials
    ) -> list[Chunk]:
        """
        Each financial statement is emitted as one chunk.

        Markdown tables are already compact and structured — splitting them
        would destroy the column header / data row relationship.
        If a statement exceeds max_chars it's still kept whole with a warning
        comment prepended so retrieval can still surface it.
        """
        statements = [
            ("financials:income_statement",  "Income Statement",        financials.income_statement_md),
            ("financials:balance_sheet",     "Balance Sheet",           financials.balance_sheet_md),
            ("financials:cash_flow",         "Cash Flow Statement",     financials.cash_flow_md),
        ]

        chunks: list[Chunk] = []
        for item_key, title, md in statements:
            if not md:
                continue
            text = f"[Financial Table – {title}]\n\n{md}"
            chunks.append(self._make_chunk(
                metadata=metadata,
                item_key=item_key,
                section_title=title,
                is_financial=True,
                chunk_index=0,
                total_chunks=1,
                text=text,
            ))
        return chunks

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        metadata: FilingMetadata,
        item_key: str,
        section_title: str,
        is_financial: bool,
        chunk_index: int,
        total_chunks: int,
        text: str,
    ) -> Chunk:
        return Chunk(
            ticker=metadata.ticker,
            accession_number=metadata.accession_number,
            filing_date=str(metadata.filing_date),
            period_of_report=str(metadata.period_of_report) if metadata.period_of_report else None,
            form=metadata.form,
            item_key=item_key,
            section_title=section_title,
            is_financial=is_financial,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            text=text,
        )
