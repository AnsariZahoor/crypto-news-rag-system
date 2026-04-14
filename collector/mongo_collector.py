
"""
Collects raw articles from MongoDB, cleans HTML content,
normalises dates, and writes a CSV ready for the ingestion pipeline.

Entry points:
    collect()          full run: MongoDB → clean → CSV
    load_from_csv()    CSV → list[dict]  (feeds run_pipeline / run_incremental)

Usage:
    from collector.mongo_collector import CollectorConfig, collect, load_from_csv
    from ingestion.pipeline import build_pipeline, run_incremental

    config  = CollectorConfig()
    csv_path = collect(config)

    raw_docs = load_from_csv(csv_path, limit=1000)
    result = run_incremental(raw_docs, pipeline)
"""

import csv
import html as html_lib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

logger = logging.getLogger(__name__)



@dataclass
class CollectorConfig:
    # MongoDB
    mongo_uri: str = field(default_factory=lambda: os.getenv("MONGO_URI", ""))
    database: str = "rag_pipeline"
    collection: str = "articles_raw"
    batch_size: int = 1000
    query_filter: dict = field(default_factory=dict)   # {} = fetch all

    # Output
    output_csv: str = "output.csv"

    # Content cleaning
    min_paragraph_len: int = 30
    boilerplate_text: str = (
        "The following article is adapted from The Block's newsletter, "
        "The Daily , which comes out on weekday afternoons."
    )

    # Date
    date_field: str = "published_at"
    date_output_fmt: str = "%Y-%m-%dT%H:%M:%SZ"

    # Fields to keep in the CSV (empty = keep all)
    fields: list[str] = field(default_factory=lambda: [
        "article_id", "title", "url", "slug",
        "tags", "published_at", "content",
    ])



def build_client(config: CollectorConfig) -> MongoClient:
    """Connect to MongoDB and verify the connection."""
    if not config.mongo_uri:
        raise ValueError("MONGO_URI is not set — check your .env file")

    client = MongoClient(config.mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    logger.info("MongoDB connected ✓")
    return client


# HTML → clean text
def _build_boilerplate_regex(text: str) -> re.Pattern:
    """Compile a whitespace-tolerant regex for a boilerplate string."""
    return re.compile(
        re.escape(text).replace("\\ ", "\\s*"),
        re.IGNORECASE,
    )


def extract_paragraphs(
    raw_html: str,
    min_length: int,
    boilerplate_re: re.Pattern,
) -> list[str]:
    """
    Parse raw HTML and return clean text from every <p> tag.

    Steps per paragraph:
        1. Extract inner text (collapses <strong>, <a>, <em>, etc.)
        2. Decode HTML entities
        3. Strip boilerplate
        4. Collapse whitespace
        5. Remove non-printable characters
        6. Drop paragraphs shorter than min_length
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    paragraphs = []

    for p_tag in soup.find_all("p"):
        text = p_tag.get_text(separator=" ")
        text = html_lib.unescape(text)
        text = boilerplate_re.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", "", text)

        if len(text) >= min_length:
            paragraphs.append(text)

    return paragraphs


def preprocess_content(
    raw_html: Any,
    min_length: int,
    boilerplate_re: re.Pattern,
) -> str:
    """
    Convert raw HTML to a clean string of paragraphs separated by double
    newlines — the format expected by RecursiveCharacterTextSplitter.

    Returns empty string if no usable paragraphs are found.
    """
    if not isinstance(raw_html, str):
        raw_html = str(raw_html) if raw_html is not None else ""

    paragraphs = extract_paragraphs(raw_html, min_length, boilerplate_re)
    return "\n\n".join(paragraphs)



def normalize_date(value: Any, output_fmt: str) -> Optional[str]:
    """
    Parse any date representation and return a UTC string.
    Returns None if the value cannot be parsed.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = pd.to_datetime(value, utc=True).to_pydatetime()
        except Exception:
            logger.warning("Unparseable date value: %r", value)
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc).strftime(output_fmt)


# Tags normalisation (handles stringified lists)
def parse_tags(raw: Any) -> str:
    """
    Return a clean comma-separated tag string.
    Handles proper lists, stringified lists, and comma strings.
    """
    import ast

    if not raw:
        return ""

    if isinstance(raw, list):
        if all(len(str(t)) <= 1 for t in raw):
            raw = "".join(str(t) for t in raw)
        else:
            return ", ".join(
                t.strip().lower() for t in raw
                if isinstance(t, str) and t.strip()
            )

    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    return ", ".join(
                        t.strip().lower() for t in parsed
                        if isinstance(t, str) and t.strip()
                    )
            except (ValueError, SyntaxError):
                pass
            raw = raw.strip("[]")

        return ", ".join(
            t.strip().strip("'\"").lower()
            for t in raw.split(",")
            if t.strip().strip("'\"")
        )

    return ""


# Document cleaner
def clean_document(
    doc: dict[str, Any],
    config: CollectorConfig,
    boilerplate_re: re.Pattern,
) -> Optional[dict[str, Any]]:
    """
    Clean a single raw MongoDB document.

    Returns None if the document should be skipped entirely.
    """
    article_id = doc.get("article_id")
    if not article_id:
        logger.warning("Skipping doc without article_id: _id=%s", doc.get("_id"))
        return None

    raw_content = doc.get("content")
    cleaned_content = preprocess_content(
        raw_content,
        config.min_paragraph_len,
        boilerplate_re,
    )

    if not cleaned_content:
        logger.warning("No usable content for article_id=%s — skipping", article_id)
        return None

    return {
        "article_id": str(article_id),
        "title": (doc.get("title") or "").strip()[:500],
        "url": doc.get("url") or "",
        "slug": doc.get("slug") or "",
        "tags": parse_tags(doc.get("tags")),
        "published_at": normalize_date(doc.get(config.date_field), config.date_output_fmt),
        "content": cleaned_content,
    }



@dataclass
class CollectResult:
    total_fetched: int = 0
    total_written: int = 0
    total_skipped: int = 0
    output_csv: str = ""

    def log_summary(self) -> None:
        logger.info(
            "Collection complete | fetched=%d | written=%d | skipped=%d | csv=%s",
            self.total_fetched,
            self.total_written,
            self.total_skipped,
            self.output_csv,
        )
        if self.total_skipped:
            logger.warning("%d documents skipped (no content or missing article_id)", self.total_skipped)


def collect(config: CollectorConfig) -> str:
    """
    Full collector run: MongoDB → clean → CSV.

    Args:
        config: CollectorConfig instance.

    Returns:
        Path to the written CSV file.
    """
    result = CollectResult(output_csv=config.output_csv)
    boilerplate_re = _build_boilerplate_regex(config.boilerplate_text)
    client = build_client(config)

    try:
        db = client[config.database]
        collection = db[config.collection]

        total = collection.count_documents(config.query_filter)
        logger.info("Found %d documents in collection", total)
        result.total_fetched = total

        cursor = collection.find(config.query_filter, batch_size=config.batch_size)

        fieldnames = config.fields or [
            "article_id", "title", "url", "slug",
            "tags", "published_at", "content",
        ]

        with open(config.output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=fieldnames,
                extrasaction="ignore",
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()

            for doc in cursor:
                cleaned = clean_document(doc, config, boilerplate_re)

                if cleaned is None:
                    result.total_skipped += 1
                    continue

                writer.writerow(cleaned)
                result.total_written += 1

                if result.total_written % 2000 == 0:
                    logger.info(
                        "Progress: %d / %d written ...",
                        result.total_written, total,
                    )

    finally:
        client.close()
        logger.info("MongoDB connection closed")

    result.log_summary()
    return config.output_csv



def load_from_csv(
    csv_path: str,
    limit: Optional[int] = None,
    sort_by_date: bool = True,
    date_field: str = "published_at",
) -> list[dict[str, Any]]:
    """
    Load the collector CSV and return a list of dicts ready for the
    ingestion pipeline (run_pipeline / run_incremental).

    Args:
        csv_path: Path to CSV written by collect().
        limit: Cap the number of returned documents. None = all.
        sort_by_date: Sort newest-first before applying limit.
        date_field: Column to sort on.

    Returns:
        list[dict] — same shape as raw_docs expected by the pipeline.
    """
    logger.info("Loading CSV: %s", csv_path)

    df = pd.read_csv(csv_path, dtype=str)

    # Normalise date column
    if date_field in df.columns:
        df[date_field] = (
            pd.to_datetime(df[date_field], format="ISO8601", utc=True)
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    if sort_by_date and date_field in df.columns:
        df.sort_values(by=date_field, ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

    if limit:
        df = df.head(limit)

    # Replace NaN with None so downstream code can use `or` guards safely
    df = df.where(pd.notna(df), None)
    print(df)

    records = df.to_dict(orient="records")
    logger.info("Loaded %d records from %s", len(records), csv_path)
    return records