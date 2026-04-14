"""
Step 1  PineconeHybridSearchRetriever   dense + sparse fusion (LangChain built-in)
Step 2  CrossEncoder                    rerank top_k candidates
Step 3  Freshness decay                 exp(-lambda * age_days)
Step 4  Score blending                  (1-w)*sigmoid(ce) + w*freshness
Step 5  Sort + truncate                 return top rerank_top_n results
"""

import os
import pytz
import time
import math
import logging

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv

from langsmith import Client
from langsmith import traceable
from langchain_core.tracers.context import collect_runs
from langchain_community.retrievers import PineconeHybridSearchRetriever
from sentence_transformers import CrossEncoder

from src.ingestion.embedder import build_embedder
from src.ingestion.vector_store import build_index, load_bm25

logger = logging.getLogger(__name__)

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

ls_client = Client()

@dataclass
class RetrieverConfig:
    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    pinecone_index: str = "crypto-news"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "embded_articles"

    # Embedding — must match what was used during ingestion
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # BM25 — must be the encoder fitted during ingestion
    bm25_path: str = "bm25_encoder.json"

    # Hybrid search
    top_k: int = 20     # candidates fetched from Pinecone
    alpha: float = 0.5    # 0.0 = sparse only, 1.0 = dense only

    # Cross-encoder
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 5 # final results returned to caller

    # Freshness
    lambda_: float = 0.1    # decay rate  (~7-day half-life for crypto news)
    freshness_weight: float = 0.3    # 0.0 = relevance only, 1.0 = recency only



@dataclass
class ScoredChunk:
    """Single retrieved chunk with all intermediate scores attached."""
    content:str
    metadata:dict
    pinecone_score:float   # raw Pinecone hybrid score
    cross_encoder_score:float   # cross-encoder logit
    freshness_score:float   # exp(-lambda * age_days)
    final_score:float   # blended final score

    # Metadata shortcuts
    @property
    def article_id(self) -> str:
        return self.metadata.get("article_id", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    @property
    def published_at(self) -> str:
        return self.metadata.get("published_at", "")

    @property
    def url(self) -> str:
        return self.metadata.get("url", "")

    @property
    def tags(self) -> str:
        return self.metadata.get("tags", "")

    @property
    def chunk_index(self) -> int:
        return int(self.metadata.get("chunk_index", 0))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dict — ready for format_context() in the LLM layer."""
        return {
            **self.metadata,
            "content": self.content,
            "score": self.final_score,
        }

    def __repr__(self) -> str:
        return (
            f"ScoredChunk(article_id={self.article_id!r}, "
            f"chunk={self.chunk_index}, "
            f"final={self.final_score:.4f}, "
            f"ce={self.cross_encoder_score:.4f}, "
            f"fresh={self.freshness_score:.4f})"
        )


@dataclass
class Retriever:
    """All initialised components. Build once at startup, reuse per query."""
    config: RetrieverConfig
    pinecone_retriever: PineconeHybridSearchRetriever
    cross_encoder: CrossEncoder



def build_retriever(config: RetrieverConfig) -> Retriever:
    """
    Initialise all retriever components from config.

    Connects to Pinecone, loads the embedding model, BM25 encoder,
    and cross-encoder — all from the same artifacts used during ingestion.

    Args:
        config: RetrieverConfig instance.

    Returns:
        Ready-to-use Retriever.
    """
    logger.info("Building retriever...")

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

    bm25 = load_bm25(config.bm25_path)

    # LangChain built-in — handles dense + sparse fusion internally
    pinecone_retriever = PineconeHybridSearchRetriever(
        embeddings=embedder,
        sparse_encoder=bm25,
        index=index,
        top_k=config.top_k,
        alpha=config.alpha,
        namespace=config.pinecone_namespace,
        text_key="content",
    )

    cross_encoder = CrossEncoder(config.cross_encoder_model)

    logger.info("Retriever built ✓")
    return Retriever(
        config=config,
        pinecone_retriever=pinecone_retriever,
        cross_encoder=cross_encoder,
    )


# Step 1 — Pinecone hybrid search (LangChain)
def _fetch_candidates(
    query:str,
    retriever:Retriever,
    top_k:int,
    alpha:float,
    filter:Optional[dict],
) -> list[Any]:
    """
    Invoke PineconeHybridSearchRetriever and return LangChain Documents.

    alpha and top_k are patched onto the retriever temporarily so
    per-call overrides work without mutating the shared config permanently.
    """
    pr = retriever.pinecone_retriever

    # Patch per-call overrides
    original_top_k = pr.top_k
    original_alpha = pr.alpha
    pr.top_k = top_k
    pr.alpha = alpha

    try:
        search_kwargs = {}
        if filter:
            search_kwargs["filter"] = filter

        docs = (
            pr.invoke(query, search_kwargs=search_kwargs)
            if search_kwargs
            else pr.invoke(query)
        )
    finally:
        # Always restore — keeps the retriever reusable
        pr.top_k = original_top_k
        pr.alpha = original_alpha

    logger.info(
        "PineconeHybridSearchRetriever returned %d docs for query: %r",
        len(docs), query[:60],
    )
    return docs


# Step 2 — Cross-encoder reranking
def _rerank(
    query: str,
    docs: list[Any],
    cross_encoder: CrossEncoder,
) -> list[tuple[Any, float]]:
    """
    Score each (query, passage) pair with the cross-encoder.

    LangChain Documents store content in doc.page_content and
    Pinecone metadata in doc.metadata.

    Returns:
        List of (doc, ce_score) tuples, unsorted.
    """
    if not docs:
        return []

    pairs  = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    logger.debug("Cross-encoder scored %d pairs", len(scores))
    return list(zip(docs, scores.tolist()))


# Step 3 — Freshness scoring
def _compute_freshness(
    published_str: Optional[str],
    now:           datetime,
    lambda_:       float,
) -> float:
    """
    Exponential time-decay: freshness = exp(-lambda_ * age_in_days)

        lambda_=0.1  → half-life ~7 days   (breaking crypto news)
        lambda_=0.01 → half-life ~70 days  (long-form analysis)

    Returns 1.0 if date is missing or unparseable.
    """
    if not published_str:
        return 1.0
    try:
        dt = datetime.fromisoformat(published_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = max((now - dt).days, 0)
        return math.exp(-lambda_ * age_days)
    except (ValueError, TypeError) as exc:
        logger.warning("Could not parse published_at=%r: %s", published_str, exc)
        return 1.0


# Step 4 — Score blending
def _sigmoid(x: float) -> float:
    """Map cross-encoder logit (any range) to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))


def _blend(ce_score: float, freshness: float, freshness_weight: float) -> float:
    """
    final = (1 - w) * sigmoid(ce_score) + w * freshness

    freshness_weight=0.0 → pure relevance (cross-encoder only)
    freshness_weight=1.0 → pure recency   (freshness only)
    """
    normalised = _sigmoid(ce_score)
    if freshness_weight == 0.0:
        return normalised
    return (1 - freshness_weight) * normalised + freshness_weight * freshness


@traceable(
    name="crypto-rag-retrieve",
    tags=["retrieval"],
)
def retrieve(
    query: str,
    retriever: Retriever,
    top_n: Optional[int] = None,
    alpha: Optional[float] = None,
    filter: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> list[ScoredChunk]:
    """
    Full retrieval pipeline for a single query.

    Args:
        query: User query string.
        retriever: Built Retriever from build_retriever().
        top_n: Override config.rerank_top_n for this call.
        alpha: Override config.alpha (0=sparse, 1=dense) for this call.
        filter: Pinecone metadata filter e.g. {"tags": {"$in": ["bitcoin"]}}.
        now: UTC reference time for freshness (defaults to current time).

    Returns:
        List of ScoredChunk sorted by final_score descending.
    """
    start = time.time()
    config  = retriever.config
    now = now or datetime.now(pytz.UTC)
    _top_n = top_n if top_n is not None else config.rerank_top_n
    _alpha = alpha if alpha is not None else config.alpha

    # ── 1. Pinecone hybrid search ────────────────────────────────────────
    docs = _fetch_candidates(query, retriever, config.top_k, _alpha, filter)

    if not docs:
        logger.warning("No candidates returned for query: %r", query)
        return []

    # ── 2. Cross-encoder reranking ───────────────────────────────────────
    reranked = _rerank(query, docs, retriever.cross_encoder)

    # ── 3. Freshness + blending → ScoredChunk ───────────────────────────
    scored: list[ScoredChunk] = []

    for doc, ce_score in reranked:
        metadata = doc.metadata
        pinecone_score = metadata.get("score", 0.0)
        freshness = _compute_freshness(
            metadata.get("published_at"),
            now,
            config.lambda_,
        )
        final = _blend(ce_score, freshness, config.freshness_weight)

        scored.append(ScoredChunk(
            content             = doc.page_content,
            metadata            = metadata,
            pinecone_score      = pinecone_score,
            cross_encoder_score = ce_score,
            freshness_score     = freshness,
            final_score         = final,
        ))

    # ── 4. Sort + truncate ───────────────────────────────────────────────
    results = sorted(scored, key=lambda c: c.final_score, reverse=True)[:_top_n]

    logger.info(
        "retrieve | query=%r | candidates=%d | returned=%d | "
        "top_score=%.4f | %.3fs",
        query[:60], len(docs), len(results),
        results[0].final_score if results else 0.0,
        time.time() - start,
    )
    return results


def retrieve_batch(
    queries:   list[str],
    retriever: Retriever,
    top_n:     Optional[int]  = None,
    filter:    Optional[dict] = None,
) -> dict[str, list[ScoredChunk]]:
    """
    Retrieve results for multiple queries.

    Uses a shared UTC timestamp so freshness scores are consistent
    across all queries in the same batch.

    Returns:
        Dict mapping each query string → list[ScoredChunk].
    """
    now     = datetime.now(pytz.UTC)
    results = {}

    for query in queries:
        results[query] = retrieve(
            query=query,
            retriever=retriever,
            top_n=top_n,
            filter=filter,
            now=now,
        )

    logger.info("retrieve_batch | queries=%d", len(queries))
    return results

if __name__ == "__main__":
    cfg = RetrieverConfig()
    retriever = build_retriever(cfg)
    results = retrieve("Where did AlphaTON invest?", retriever)

    raw_docs = [c.to_dict() for c in results]   # feed into LLM layer
    print(raw_docs)