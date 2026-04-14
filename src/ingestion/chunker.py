import ast
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


MIN_CONTENT_LENGTH = 50   # chars — skip articles shorter than this
TITLE_MAX_LENGTH   = 500  # match your metadata cap


# Date normalisation (shared with retriever.py)
def normalize_date(value: Any) -> Optional[str]:
    """
    Always return a UTC ISO-8601 string or None.
    Handles: datetime objects, strings with/without timezone.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            logger.warning("Unparseable published_at value: %r", value)
            return None
    return None


def parse_tags(raw: Any) -> list[str]:
    """
    Safely parse tags regardless of how they arrive from the DB.

    Handles:
        - Proper list:        ["analyst-reports", "bitfinex"]
        - String of a list:   "['analyst-reports', 'bitfinex']"
        - Comma string:       "analyst-reports, bitfinex"
        - Char-split garbage: "[, ', a, n, a, l, ...]"
        - None / empty
    """
    if not raw:
        return []

    # Already a clean list
    if isinstance(raw, list):
        # Guard against the char-split case: ["a","n","a","l","y","s","t"]
        # Real tags are words/slugs, not single characters
        if all(len(str(t)) <= 1 for t in raw):
            # Every element is 1 char → the string was iterated, not parsed
            # Reconstruct by joining and re-parsing
            raw = "".join(str(t) for t in raw)
        else:
            return [t.strip().lower() for t in raw if isinstance(t, str) and t.strip()]

    if isinstance(raw, str):
        raw = raw.strip()

        # Looks like a Python list literal → parse it safely
        if raw.startswith("["):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    return [t.strip().lower() for t in parsed if isinstance(t, str) and t.strip()]
            except (ValueError, SyntaxError):
                pass

            # ast failed — strip brackets and split on commas
            raw = raw.strip("[]")

        # Plain comma-separated string: "analyst-reports, bitfinex"
        return [t.strip().strip("'\"").lower() for t in raw.split(",") if t.strip().strip("'\"")]

    return []

# Text assembly  (only content that should be embedded)
def build_full_text(doc: dict[str, Any]) -> str:
    """
    Assemble embeddable text from a raw document.

    Title is prepended as a chunk header so every chunk retains
    article context even after splitting.
    """
    title   = (doc.get("title") or "").strip()
    content = (doc.get("content") or "").strip()

    parts = []
    if title:
        parts.append(f"Title: {title}")  # chunk header — aids retrieval
    if content:
        parts.append(content)

    return "\n\n".join(parts)


# Splitter factory
def build_splitter(
    chunk_size: int    = 512,
    chunk_overlap: int = 64,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# Metadata builder
def build_metadata(
    doc: dict[str, Any],
    chunk_index: int,
    chunk_total: int,
) -> dict[str, Any]:
    return {
        "article_id": str(doc.get("article_id") or ""),
        "title": (doc.get("title") or "").strip()[:TITLE_MAX_LENGTH],
        "url": doc.get("url") or "",
        "slug": doc.get("slug") or "",
        "tags": ", ".join(parse_tags(doc.get("tags", []))),
        "published_at": normalize_date(doc.get("published_at")),
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,   # ← new: e.g. 2 of 5
    }


def split_document(
    doc: dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
) -> tuple[Optional[str], list[str]]:
    """
    Validate and split a single document into text chunks.

    Returns:
        (article_id, chunks) — article_id is None if doc should be skipped.
    """
    article_id = doc.get("article_id")
    if article_id is None:
        logger.warning("Skipping document without article_id: mongo_id=%s", doc.get("_id"))
        return None, []

    content = (doc.get("content") or "").strip()
    if len(content) < MIN_CONTENT_LENGTH:
        logger.warning(
            "Skipping article_id=%s — content too short (%d chars)",
            article_id, len(content),
        )
        return None, []

    full_text = build_full_text(doc)
    chunks = splitter.split_text(full_text)

    logger.debug("article_id=%s → %d chunks", article_id, len(chunks))
    return str(article_id), chunks



def chunk_document(
    doc: dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
) -> list[dict[str, Any]]:
    """
    Convert one raw article dict into a list of chunk records.

    Each record shape:
        {
            "id":       "article_id_chunkindex",   e.g. "art_001_0"
            "text":     "...",                      text to embed
            "metadata": { ...article fields... }
        }
    """
    article_id, chunks = split_document(doc, splitter)
    if article_id is None:
        return []

    chunk_total = len(chunks)
    records: list[dict[str, Any]] = []

    for chunk_index, chunk_text in enumerate(chunks):
        records.append({
            "id": f"{article_id}_{chunk_index}",
            "text": chunk_text,
            "metadata": build_metadata(doc, chunk_index, chunk_total),
        })

    return records



def chunk_documents(
    docs: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
) -> list[dict[str, Any]]:
    """
    Chunk a batch of raw article dicts.

    Args:
        docs:     List of raw documents from the collector.
        splitter: RecursiveCharacterTextSplitter instance from build_splitter().

    Returns:
        Flat list of chunk records ready for embedding + vector storage.
    """
    if not docs:
        logger.warning("chunk_documents called with empty docs list")
        return []

    records: list[dict[str, Any]] = []

    for i, doc in enumerate(docs):
        doc_records = chunk_document(doc, splitter)
        records.extend(doc_records)

        # Progress log every 100 docs — useful for large ingestion runs
        if (i + 1) % 100 == 0:
            logger.info("Chunked %d / %d docs → %d chunks so far", i + 1, len(docs), len(records))

    logger.info(
        "chunk_documents complete | docs=%d | chunks=%d | skipped=%d",
        len(docs),
        len(records),
        len(docs) - len({r["metadata"]["article_id"] for r in records}),
    )
    return records