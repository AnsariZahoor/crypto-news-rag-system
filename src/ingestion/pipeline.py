"""
Wires together:
    chunker.py      → split raw articles into chunk records
    embedder.py     → embed chunks with all-MiniLM-L6-v2
    vector_store.py → upsert dense + sparse vectors to Pinecone

Entry points:
    run_pipeline()        full pipeline for a batch of raw docs
    run_incremental()     upsert only new articles (skips existing IDs)

Usage:
    from ingestion.pipeline import build_pipeline, run_pipeline

    pipeline = build_pipeline(config)
    result   = run_pipeline(raw_docs, pipeline)
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv
from collector.mongo_collector import CollectorConfig, collect, load_from_csv

from src.ingestion.chunker import (
    build_splitter,
    chunk_documents,
)
from src.ingestion.embedder import (
    build_embedder,
    embed_chunks,
    validate_embeddings,
)
from src.ingestion.vector_store import (
    build_index,
    build_bm25,
    load_bm25,
    save_bm25,
    upsert_records,
    index_stats,
)

logger = logging.getLogger(__name__)

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

@dataclass
class PipelineConfig:
    """
    All tuneable settings in one place.
    Instantiate once and pass to build_pipeline().
    """
    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    pinecone_index: str = "crypto-news"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "embded_articles"

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_dim: int = 384
    embed_batch_size: int = 64

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # BM25
    bm25_path: str = "bm25_encoder.json"
    bm25_fit_on_first_run: bool = True   # auto-fit if no saved encoder found

    # Upsert
    upsert_batch_size: int = 100

    # Alpha for hybrid search (stored for reference — used at query time)
    default_alpha: float = 0.5



@dataclass
class Pipeline:
    """
    Holds all initialised components.
    Built once at startup, reused across multiple run_pipeline() calls.
    """
    config: PipelineConfig
    splitter: Any
    embedder: Any
    index: Any
    bm25: Optional[Any] = None   # None until first corpus is available



def build_pipeline(config: PipelineConfig) -> Pipeline:
    """
    Initialise all components from config.

    Loads a saved BM25 encoder if one exists at config.bm25_path.
    BM25 fitting is deferred to run_pipeline() if no saved encoder is found.

    Args:
        config: PipelineConfig instance.

    Returns:
        Ready-to-use Pipeline instance.
    """
    logger.info("Building ingestion pipeline...")

    splitter = build_splitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    embedder = build_embedder(
        model_name=config.embedding_model,
        device=config.embedding_device,
    )

    index = build_index(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index,
        cloud=config.pinecone_cloud,
        region=config.pinecone_region,
    )

    # Try loading a pre-fitted BM25 encoder
    bm25 = None
    try:
        bm25 = load_bm25(config.bm25_path)
        logger.info("Loaded existing BM25 encoder from %s", config.bm25_path)
    except FileNotFoundError:
        logger.info(
            "No BM25 encoder found at %s — will fit on first run",
            config.bm25_path,
        )

    logger.info("Pipeline built ✓")
    return Pipeline(
        config=config,
        splitter=splitter,
        embedder=embedder,
        index=index,
        bm25=bm25,
    )



@dataclass
class PipelineResult:
    total_docs: int = 0
    total_chunks: int = 0
    total_embedded: int = 0
    total_upserted: int = 0
    total_skipped: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.total_upserted > 0 and not self.errors

    def log_summary(self) -> None:
        logger.info(
            "Pipeline result | docs=%d | chunks=%d | embedded=%d "
            "| upserted=%d | skipped=%d | %.1fs",
            self.total_docs,
            self.total_chunks,
            self.total_embedded,
            self.total_upserted,
            self.total_skipped,
            self.elapsed_seconds,
        )
        if self.errors:
            for err in self.errors:
                logger.error("Pipeline error: %s", err)



def run_pipeline(
    raw_docs: list[dict[str, Any]],
    pipeline: Pipeline,
) -> PipelineResult:
    """
    Full ingestion pipeline: chunk → embed → upsert.

    Args:
        raw_docs: List of raw article dicts from your collector.
                  Each must have: article_id, title, content,
                  url, slug, tags, published_at.
        pipeline: Built Pipeline instance from build_pipeline().

    Returns:
        PipelineResult with counts and timing.
    """
    result = PipelineResult()
    start  = time.time()

    if not raw_docs:
        logger.warning("run_pipeline called with empty docs list")
        return result

    result.total_docs = len(raw_docs)
    logger.info("Starting ingestion for %d documents", result.total_docs)

    # ── Step 1: Chunk ────────────────────────────────────────────────────
    logger.info("Step 1/3 — Chunking...")
    try:
        records = chunk_documents(raw_docs, pipeline.splitter)
    except Exception as exc:
        msg = f"Chunking failed: {exc}"
        logger.exception(msg)
        result.errors.append(msg)
        return result

    if not records:
        logger.warning("No chunks produced from %d docs — nothing to ingest", result.total_docs)
        return result

    result.total_chunks = len(records)
    logger.info("Chunking complete — %d chunks from %d docs", result.total_chunks, result.total_docs)

    # ── Step 2: Embed ────────────────────────────────────────────────────
    logger.info("Step 2/3 — Embedding...")
    try:
        embedded = embed_chunks(
            records,
            pipeline.embedder,
            batch_size=pipeline.config.embed_batch_size,
        )
    except Exception as exc:
        msg = f"Embedding failed: {exc}"
        logger.exception(msg)
        result.errors.append(msg)
        return result

    if not validate_embeddings(embedded, expected_dim=pipeline.config.embedding_dim):
        msg = "Embedding validation failed — aborting upsert"
        logger.error(msg)
        result.errors.append(msg)
        return result

    result.total_embedded = len(embedded)
    logger.info("Embedding complete — %d vectors", result.total_embedded)

    # ── Step 3: Fit or load BM25 ─────────────────────────────────────────
    if pipeline.bm25 is None:
        if pipeline.config.bm25_fit_on_first_run:
            logger.info("Fitting BM25 on current corpus...")
            try:
                pipeline.bm25 = build_bm25(embedded)
                save_bm25(pipeline.bm25, pipeline.config.bm25_path)
            except Exception as exc:
                msg = f"BM25 fitting failed: {exc}"
                logger.exception(msg)
                result.errors.append(msg)
                return result
        else:
            msg = "BM25 encoder not available and bm25_fit_on_first_run=False — aborting"
            logger.error(msg)
            result.errors.append(msg)
            return result

    # ── Step 4: Upsert ───────────────────────────────────────────────────
    logger.info("Step 3/3 — Upserting to Pinecone...")
    try:
        upsert_result = upsert_records(
            embedded,
            pipeline.index,
            pipeline.bm25,
            namespace=pipeline.config.pinecone_namespace,
            batch_size=pipeline.config.upsert_batch_size,
        )
    except Exception as exc:
        msg = f"Upsert failed: {exc}"
        logger.exception(msg)
        result.errors.append(msg)
        return result

    result.total_upserted = upsert_result["upserted"]
    result.total_skipped  = upsert_result["skipped"]
    result.elapsed_seconds = time.time() - start

    result.log_summary()
    return result


# Incremental ingestion — skip already-indexed articles
def run_incremental(
    raw_docs: list[dict[str, Any]],
    pipeline: Pipeline,
) -> PipelineResult:
    """
    Ingest only articles not already present in the index.

    Checks Pinecone for existing vector IDs (article_id_0 of each doc)
    and filters out docs that are already indexed.

    Args:
        raw_docs: Full list of raw docs — may contain already-indexed ones.
        pipeline: Built Pipeline instance.

    Returns:
        PipelineResult. total_skipped includes pre-existing docs.
    """
    if not raw_docs:
        return PipelineResult()

    logger.info("Incremental check: %d candidate docs", len(raw_docs))

    # Check first chunk ID of each doc (chunk_index=0 always exists if doc was ingested)
    candidate_ids = [f"{str(doc.get('article_id', ''))}_0" for doc in raw_docs]

    try:
        fetch_response = pipeline.index.fetch(
            ids=candidate_ids,
            namespace=pipeline.config.pinecone_namespace,
        )
        existing_ids = set(fetch_response.vectors.keys())
    except Exception as exc:
        logger.warning("Could not fetch existing IDs from Pinecone: %s — ingesting all", exc)
        existing_ids = set()

    new_docs = [
        doc for doc, cid in zip(raw_docs, candidate_ids)
        if cid not in existing_ids
    ]

    already_indexed = len(raw_docs) - len(new_docs)
    logger.info(
        "Incremental filter | total=%d | already_indexed=%d | new=%d",
        len(raw_docs), already_indexed, len(new_docs),
    )

    if not new_docs:
        logger.info("All docs already indexed — nothing to do")
        result = PipelineResult(
            total_docs=len(raw_docs),
            total_skipped=already_indexed,
        )
        return result

    result = run_pipeline(new_docs, pipeline)
    result.total_skipped += already_indexed
    return result


if __name__ == "__main__":
    # ── Step 1: Collect & clean from MongoDB → CSV ──────────────────────────
    collector_config = CollectorConfig()
    csv_path = collect(collector_config)

    # ── Step 2: Load CSV → pipeline-ready list ──────────────────────────────
    raw_docs = load_from_csv(csv_path, limit=1000, sort_by_date=True)    

    # # ── Step 3: Ingest into Pinecone ────────────────────────────────────────
    config = PipelineConfig()
    pipeline = build_pipeline(config)   # loads model + connects to Pinecone

    result = run_pipeline(raw_docs, pipeline)

    print(result.total_docs)      # 500
    print(result.total_chunks)    # 1,243
    print(result.total_upserted)  # 1,243
    print(result.elapsed_seconds) # 47.2
    print(result.success)         # True