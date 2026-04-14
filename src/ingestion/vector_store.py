"""
Pinecone vector storage layer with hybrid search (dense + sparse).

Dense  vectors : all-MiniLM-L6-v2  (384-dim, from embedder.py)
Sparse vectors : BM25               (keyword matching via pinecone-text)
"""

import logging
import time
from typing import Any, Optional

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

logger = logging.getLogger(__name__)


DENSE_DIM = 384 # all-MiniLM-L6-v2 output dimension
UPSERT_BATCH = 100 # Pinecone recommended max per upsert call
DEFAULT_TOP_K = 20 # candidates fetched before reranking
DEFAULT_ALPHA = 0.5 # 0.0 = pure sparse, 1.0 = pure dense


def build_index(
    api_key: str,
    index_name: str,
    cloud: str = "aws",
    region: str = "us-east-1",
) -> Any:
    """
    Connect to an existing Pinecone index or create it if missing.

    Hybrid search requires metric="dotproduct" — do not change this.
    cosine similarity is NOT supported for sparse-dense hybrid in Pinecone.

    Args:
        api_key:    Pinecone API key.
        index_name: Name of the index.
        cloud:      Cloud provider — "aws" | "gcp" | "azure".
        region:     Cloud region.

    Returns:
        Pinecone Index object.
    """
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing:
        logger.info("Creating Pinecone index: %s", index_name)
        pc.create_index(
            name=index_name,
            dimension=DENSE_DIM,
            metric="dotproduct",        # required for hybrid search
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait until index is ready
        while not pc.describe_index(index_name).status["ready"]:
            logger.info("Waiting for index to be ready...")
            time.sleep(2)

        logger.info("Index created ✓")
    else:
        logger.info("Connected to existing index: %s", index_name)

    return pc.Index(index_name)


# BM25 sparse encoder
def build_bm25(
    records: list[dict[str, Any]],
    text_key: str = "text",
) -> BM25Encoder:
    """
    Fit a BM25 encoder on your corpus.

    IMPORTANT: fit on the full corpus once, then save and reuse.
    Re-fitting on a subset changes the IDF weights and breaks
    consistency between old and new vectors.

    Args:
        records:  Chunk records (output of chunk_documents / embed_chunks).
        text_key: Key holding the text to fit on.

    Returns:
        Fitted BM25Encoder instance.
    """
    texts = [r[text_key] for r in records if r.get(text_key)]
    logger.info("Fitting BM25 on %d documents...", len(texts))
    encoder = BM25Encoder()
    encoder.fit(texts)
    logger.info("BM25 fitted ✓")
    return encoder


def save_bm25(encoder: BM25Encoder, path: str = "bm25_encoder.json") -> None:
    """Persist BM25 params — must be saved and reused across ingestion runs."""
    encoder.dump(path)
    logger.info("BM25 encoder saved to %s", path)


def load_bm25(path: str = "bm25_encoder.json") -> BM25Encoder:
    """Load a previously fitted BM25 encoder."""
    encoder = BM25Encoder()
    encoder.load(path)
    logger.info("BM25 encoder loaded from %s", path)
    return encoder


# Record → Pinecone vector converter
def _to_pinecone_vector(
    record: dict[str, Any],
    bm25: BM25Encoder,
    text_key: str = "text",
) -> Optional[dict[str, Any]]:
    """
    Convert one embedded chunk record to a Pinecone upsert-ready dict.

    Pinecone hybrid vector shape:
        {
            "id":       str,
            "values":   [float, ...],        # dense vector
            "sparse_values": {
                "indices": [int, ...],
                "values":  [float, ...]
            },
            "metadata": { ... }
        }
    """
    dense_vec = record.get("embedding")
    text = record.get(text_key, "")

    if not dense_vec:
        logger.warning("Record %s missing embedding — skipped", record.get("id"))
        return None

    if not text:
        logger.warning("Record %s missing text — skipped", record.get("id"))
        return None

    # Encode sparse vector for this chunk
    sparse = bm25.encode_documents([text])[0]

    # Guard: skip if sparse encoding produced no terms
    if not sparse.get("indices"):
        logger.warning("Record %s produced empty sparse vector — skipped", record.get("id"))
        return None

    # Merge text into metadata so PineconeHybridSearchRetriever can find it
    metadata = {
        **record.get("metadata", {}),
        "content": text,            # ← this was missing
    }

    return {
        "id": record["id"],
        "values": dense_vec,
        "sparse_values": {
            "indices": sparse["indices"],
            "values": sparse["values"],
        },
        "metadata": metadata,
    }


def upsert_records(
    records: list[dict[str, Any]],
    index: Any,
    bm25: BM25Encoder,
    namespace: str = "embded_articles",
    batch_size: int = UPSERT_BATCH,
    text_key: str = "text",
) -> dict[str, int]:
    """
    Convert and upsert embedded chunk records into Pinecone.

    Args:
        records:    Output of embed_chunks() — must have "embedding" field.
        index:      Pinecone Index from build_index().
        bm25:       Fitted BM25Encoder from build_bm25() or load_bm25().
        namespace:  Pinecone namespace (use for multi-tenant isolation).
        batch_size: Vectors per upsert call (max 100 recommended by Pinecone).
        text_key:   Key holding chunk text inside each record.

    Returns:
        {"upserted": int, "skipped": int}
    """
    if not records:
        logger.warning("upsert_records called with empty records list")
        return {"upserted": 0, "skipped": 0}

    # Convert all records to Pinecone format
    vectors  = []
    skipped  = 0

    for record in records:
        vec = _to_pinecone_vector(record, bm25, text_key)
        if vec:
            vectors.append(vec)
        else:
            skipped += 1

    if not vectors:
        logger.error("No valid vectors to upsert after conversion")
        return {"upserted": 0, "skipped": skipped}

    # Upsert in batches
    total_upserted = 0
    batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]

    logger.info(
        "Upserting %d vectors in %d batches to namespace=%r",
        len(vectors), len(batches), namespace or "default",
    )

    for batch_num, batch in enumerate(batches, 1):
        for attempt in range(1, 4):
            try:
                response = index.upsert(
                    vectors=batch,
                    namespace=namespace,
                )
                total_upserted += response.upserted_count
                logger.info(
                    "Batch %d/%d upserted (%d vectors)",
                    batch_num, len(batches), response.upserted_count,
                )
                break

            except Exception as exc:
                logger.warning(
                    "Upsert batch %d failed (attempt %d/3): %s",
                    batch_num, attempt, exc,
                )
                if attempt < 3:
                    time.sleep(2 * attempt)
                else:
                    logger.error("Batch %d permanently failed", batch_num)
                    skipped += len(batch)

        if batch_num % 10 == 0:
            logger.info("Progress: %d/%d batches upserted", batch_num, len(batches))

    logger.info(
        "upsert_records complete | upserted=%d | skipped=%d",
        total_upserted, skipped,
    )
    return {"upserted": total_upserted, "skipped": skipped}



# Hybrid search (query time)
def hybrid_search(
    query: str,
    index: Any,
    embedder: Any,
    bm25: BM25Encoder,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    namespace: str = "",
    filter: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """
    Run a hybrid (dense + sparse) query against Pinecone.

    Alpha controls the dense/sparse balance:
        alpha=1.0 → pure semantic (dense only)
        alpha=0.5 → balanced hybrid        ← recommended starting point
        alpha=0.0 → pure keyword (BM25 only)

    Args:
        query:     User query string.
        index:     Pinecone Index from build_index().
        embedder:  Embedder from embedder.build_embedder().
        bm25:      Fitted BM25Encoder from load_bm25().
        top_k:     Number of results to return.
        alpha:     Dense/sparse weighting (0.0–1.0).
        namespace: Pinecone namespace.
        filter:    Optional metadata filter e.g. {"tags": {"$in": ["bitcoin"]}}.

    Returns:
        List of match dicts with id, score, and metadata.
    """
    # Dense vector for query
    dense_vec = embedder.embed_query(query)

    # Sparse vector for query
    sparse = bm25.encode_queries([query])[0]

    # Scale vectors by alpha (Pinecone hybrid weighting)
    scaled_dense  = [v * alpha       for v in dense_vec]
    scaled_sparse = {
        "indices": sparse["indices"],
        "values":  [v * (1 - alpha) for v in sparse["values"]],
    }

    query_kwargs = dict(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    if filter:
        query_kwargs["filter"] = filter

    response = index.query(**query_kwargs)

    results = [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata,
        }
        for match in response.matches
    ]

    logger.info(
        "hybrid_search | query=%r | alpha=%.1f | results=%d | top_score=%.4f",
        query[:60], alpha,
        len(results),
        results[0]["score"] if results else 0.0,
    )
    return results


# Index stats (useful for monitoring)
def index_stats(index: Any) -> dict[str, Any]:
    """Fetch and log current index statistics."""
    stats = index.describe_index_stats()
    logger.info(
        "Index stats | total_vectors=%d | namespaces=%s | dimension=%d",
        stats.total_vector_count,
        list(stats.namespaces.keys()),
        stats.dimension,
    )
    return stats