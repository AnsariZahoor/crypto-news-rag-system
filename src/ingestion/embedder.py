"""
Responsibilities:
    - Initialise the embedding model (HuggingFace or OpenAI)
    - Embed a batch of chunk records from chunker.py
    - Retry failed batches
    - Return records enriched with their embedding vector

Usage:
    from ingestion.embedder import build_embedder, embed_chunks

    embedder = build_embedder()
    embedded = embed_chunks(chunk_records, embedder)
"""

import logging
import time
from typing import Any, Optional, Protocol

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


# Constants
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64      # safe for most GPUs / free-tier APIs
MAX_RETRIES = 3
RETRY_DELAY = 2.0     # seconds between retries


# Embedder protocol — lets you swap HuggingFace ↔ OpenAI without changing the rest of the pipeline
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...



def build_embedder(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    normalize: bool = True,
) -> HuggingFaceEmbeddings:
    """
    Build a HuggingFace sentence-transformer embedder.

    Args:
        model_name: HuggingFace model ID.
        device:     "cpu" | "cuda" | "mps"
        normalize:  L2-normalise vectors (required for cosine similarity search).

    Returns:
        LangChain HuggingFaceEmbeddings instance.
    """
    logger.info("Loading embedding model: %s on %s", model_name, device)

    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
    )

    logger.info("Embedding model loaded ✓")
    return embedder


# Core: embed one batch with retry
def _embed_batch(
    texts: list[str],
    embedder: Embedder,
    batch_num: int,
) -> Optional[list[list[float]]]:
    """
    Embed a single batch of texts with retry logic.

    Returns:
        List of vectors, or None if all retries fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vectors = embedder.embed_documents(texts)
            logger.debug("Batch %d embedded (%d texts)", batch_num, len(texts))
            return vectors

        except Exception as exc:
            logger.warning(
                "Batch %d failed (attempt %d/%d): %s",
                batch_num, attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)   # exponential-ish back-off

    logger.error("Batch %d permanently failed after %d attempts", batch_num, MAX_RETRIES)
    return None


# Public: embed a full list of chunk records
def embed_chunks(
    records: list[dict[str, Any]],
    embedder: Embedder,
    batch_size: int = DEFAULT_BATCH_SIZE,
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """
    Embed all chunk records produced by chunker.chunk_documents().

    Each input record:
        {
            "id":       "art_001_0",
            "text":     "Title: Bitcoin...\n\nBitcoin climbed...",
            "metadata": { ... }
        }

    Each output record adds:
        {
            ...,
            "embedding": [0.021, -0.134, ...]   # float list, ready for vector DB
        }

    Args:
        records:    Chunk records from chunker.chunk_documents().
        embedder:   Embedder instance from build_embedder().
        batch_size: Number of texts per embedding call.
        text_key:   Key in each record that holds the text to embed.

    Returns:
        Records with "embedding" field added. Failed batches are skipped
        with a warning — the pipeline does not crash on partial failure.
    """
    if not records:
        logger.warning("embed_chunks called with empty records list")
        return []

    embedded: list[dict[str, Any]] = []
    skipped  = 0
    total    = len(records)
    batches  = [records[i:i + batch_size] for i in range(0, total, batch_size)]

    logger.info(
        "Embedding %d chunks in %d batches (batch_size=%d)",
        total, len(batches), batch_size,
    )

    start = time.time()

    for batch_num, batch in enumerate(batches, 1):
        texts   = [r[text_key] for r in batch]
        vectors = _embed_batch(texts, embedder, batch_num)

        if vectors is None:
            # Batch permanently failed — skip, log, continue
            logger.error(
                "Skipping batch %d (%d records): %s ... %s",
                batch_num, len(batch),
                batch[0]["id"], batch[-1]["id"],
            )
            skipped += len(batch)
            continue

        for record, vector in zip(batch, vectors):
            embedded.append({**record, "embedding": vector})

        # Progress log every 10 batches
        if batch_num % 10 == 0:
            elapsed = time.time() - start
            rate    = (batch_num * batch_size) / elapsed
            logger.info(
                "Progress: %d/%d batches | %.0f chunks/s",
                batch_num, len(batches), rate,
            )

    elapsed = time.time() - start

    logger.info(
        "embed_chunks complete | total=%d | embedded=%d | skipped=%d | %.1fs",
        total, len(embedded), skipped, elapsed,
    )

    if skipped:
        logger.warning("%d chunks were skipped due to embedding failures", skipped)

    return embedded


# Utility: embed a single query at retrieval time
def embed_query(text: str, embedder: Embedder) -> list[float]:
    """
    Embed a single query string.
    Used at retrieval time, not ingestion time.

    Args:
        text:     Query string.
        embedder: Embedder instance.

    Returns:
        Single embedding vector as a float list.
    """
    vector = embedder.embed_query(text)
    logger.debug("Query embedded: %r → dim=%d", text[:60], len(vector))
    return vector


# Utility: validate embedding output
def validate_embeddings(
    embedded: list[dict[str, Any]],
    expected_dim: Optional[int] = None,
) -> bool:
    """
    Sanity-check embedded records before pushing to vector DB.

    Checks:
        - Every record has an "embedding" key
        - All vectors have the same dimension
        - No NaN or None values
        - Optionally assert a specific dimension

    Returns:
        True if all checks pass, False otherwise.
    """
    if not embedded:
        logger.warning("validate_embeddings: empty list")
        return False

    dims = set()

    for i, record in enumerate(embedded):
        vec = record.get("embedding")

        if vec is None:
            logger.error("Record %d (%s) missing embedding", i, record.get("id"))
            return False

        if any(v is None or v != v for v in vec):   # NaN check: v != v is True for NaN
            logger.error("Record %d (%s) contains NaN/None in vector", i, record.get("id"))
            return False

        dims.add(len(vec))

    if len(dims) > 1:
        logger.error("Inconsistent embedding dimensions found: %s", dims)
        return False

    actual_dim = dims.pop()

    if expected_dim and actual_dim != expected_dim:
        logger.error("Expected dim=%d but got dim=%d", expected_dim, actual_dim)
        return False

    logger.info("Embedding validation passed | records=%d | dim=%d", len(embedded), actual_dim)
    return True