"""
Embedding client using Voyage AI.

Wraps the Voyage AI Python SDK to:
  - Batch chunks efficiently (API max: 128 texts per request)
  - Separate query embeddings (input_type="query") from document
    embeddings (input_type="document") — Voyage AI optimises internally
  - Attach embeddings back onto Chunk objects
  - Retry transiently on rate-limit / 5xx responses

Model choices:
  voyage-3-lite  512-dim  32K ctx  $0.02/1M tokens  (default, good for RAG)
  voyage-3       1024-dim 32K ctx  $0.06/1M tokens  (higher quality, 3× cost)

For pgvector in Supabase:
  - voyage-3-lite → vector(512)
  - voyage-3      → vector(1024)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import voyageai

from lib.chunker import Chunk

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOYAGE_LITE_MODEL = "voyage-3-lite"   # 512-dim, recommended default
VOYAGE_FULL_MODEL = "voyage-3"        # 1024-dim, higher quality

MAX_BATCH_SIZE = 128    # Voyage AI hard limit per request
MAX_RETRIES = 3
RETRY_DELAY = 2.0       # seconds, doubled on each retry


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedChunk:
    """A Chunk with its embedding vector attached."""
    chunk: Chunk
    embedding: list[float]
    model: str

    @property
    def dimensions(self) -> int:
        return len(self.embedding)


@dataclass
class EmbeddingStats:
    """Counters returned after a batch embed call."""
    total_chunks: int = 0
    total_tokens: int = 0
    batches: int = 0
    errors: int = 0

    def __str__(self) -> str:
        return (
            f"{self.total_chunks} chunks embedded in {self.batches} batches, "
            f"~{self.total_tokens:,} tokens, {self.errors} errors"
        )


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class VoyageEmbedder:
    """
    Embeds Chunk objects using the Voyage AI API.

    Usage
    -----
    >>> embedder = VoyageEmbedder()
    >>> embedded, stats = embedder.embed_chunks(chunks)
    >>> print(stats)
    128 chunks embedded in 1 batches, ~48,000 tokens, 0 errors

    >>> query_vec = embedder.embed_query("What was Apple's revenue in 2024?")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = VOYAGE_LITE_MODEL,
    ):
        """
        Parameters
        ----------
        api_key : str, optional
            Voyage AI API key. Falls back to VOYAGE_API_KEY env var.
        model : str
            Voyage model name. Use VOYAGE_LITE_MODEL (default) or VOYAGE_FULL_MODEL.
        """
        _key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not _key:
            raise ValueError(
                "Voyage AI API key required. Pass api_key= or set VOYAGE_API_KEY."
            )
        self.model = model
        self._client = voyageai.Client(api_key=_key)
        log.info("VoyageEmbedder ready  model=%s", model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = False,
    ) -> tuple[list[EmbeddedChunk], EmbeddingStats]:
        """
        Embed a list of Chunk objects.

        Texts sent to the API use Chunk.text_for_embedding() which prepends
        a context prefix (ticker, form, section) before the content.
        input_type="document" tells Voyage to optimise for storage/retrieval.

        Parameters
        ----------
        chunks       : chunks to embed
        show_progress: print a progress line every batch

        Returns
        -------
        (embedded_chunks, stats)
        embedded_chunks preserves the same order as the input list.
        Chunks that fail after retries are skipped and counted in stats.errors.
        """
        stats = EmbeddingStats(total_chunks=len(chunks))
        results: list[EmbeddedChunk] = []

        batches = _batch(chunks, MAX_BATCH_SIZE)
        for i, batch in enumerate(batches):
            if show_progress:
                print(f"  Embedding batch {i+1}/{len(batches)}  ({len(batch)} chunks)…")

            texts = [c.text_for_embedding() for c in batch]
            vectors, used_tokens = self._embed_with_retry(texts, input_type="document")

            if vectors is None:
                # All retries failed for this batch
                log.error("Batch %d failed after retries, skipping %d chunks", i, len(batch))
                stats.errors += len(batch)
                continue

            stats.batches += 1
            stats.total_tokens += used_tokens or _estimate_tokens(texts)

            for chunk, vector in zip(batch, vectors):
                results.append(EmbeddedChunk(chunk=chunk, embedding=vector, model=self.model))

        log.info("embed_chunks done: %s", stats)
        return results, stats

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for similarity search.

        Uses input_type="query" — Voyage applies different internal weighting
        compared to document embeddings, improving retrieval precision.
        """
        vectors, _ = self._embed_with_retry([query], input_type="query")
        if vectors is None:
            raise RuntimeError("Query embedding failed after retries.")
        return vectors[0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_with_retry(
        self,
        texts: list[str],
        input_type: str,
    ) -> tuple[Optional[list[list[float]]], Optional[int]]:
        """
        Call the Voyage API with exponential backoff on transient errors.

        Returns (embeddings, total_tokens) or (None, None) on total failure.
        """
        delay = RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.embed(
                    texts=texts,
                    model=self.model,
                    input_type=input_type,
                    truncation=True,   # silently truncate texts that exceed context
                )
                embeddings = [e for e in response.embeddings]
                tokens = getattr(response, "total_tokens", None)
                return embeddings, tokens

            except Exception as exc:
                msg = str(exc)
                is_rate_limit = "429" in msg or "rate" in msg.lower()
                is_server_err = "5" in msg[:3] or "server" in msg.lower()

                if attempt < MAX_RETRIES and (is_rate_limit or is_server_err):
                    log.warning(
                        "Voyage API error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    log.error("Voyage API error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                    return None, None

        return None, None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _batch(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _estimate_tokens(texts: list[str]) -> int:
    """Rough token count when the API doesn't return usage."""
    return int(sum(len(t) for t in texts) / 3.8)
