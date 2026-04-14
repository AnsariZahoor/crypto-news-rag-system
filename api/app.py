"""
Endpoints:
    POST /query          single question → RAGResponse
    POST /query/batch    multiple questions → list[RAGResponse]
    GET  /health         liveness check
    GET  /index/stats    Pinecone index statistics

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llm_client import LLMConfig, LLMLayer, RAGResponse, build_llm_layer, answer_async
from src.retrieval.retriever import RetrieverConfig, Retriever, ScoredChunk, build_retriever, retrieve, retrieve_batch
from src.ingestion.vector_store import index_stats

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class AppState:
    retriever: Retriever
    llm: LLMLayer


state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build all components once on startup, clean up on shutdown."""
    logger.info("Starting up crypto RAG API...")

    retriever_config = RetrieverConfig(
        pinecone_api_key = os.getenv("PINECONE_API_KEY", ""),
        pinecone_index = os.getenv("PINECONE_INDEX", "crypto-news"),
        pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region = os.getenv("PINECONE_REGION", "us-east-1"),
        pinecone_namespace = os.getenv("PINECONE_NAMESPACE", ""),
        bm25_path = os.getenv("BM25_PATH", "bm25_encoder.json"),
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu"),
        top_k = int(os.getenv("RETRIEVER_TOP_K", "20")),
        rerank_top_n = int(os.getenv("RETRIEVER_TOP_N", "5")),
        alpha = float(os.getenv("RETRIEVER_ALPHA", "0.5")),
        lambda_ = float(os.getenv("RETRIEVER_LAMBDA", "0.1")),
        freshness_weight = float(os.getenv("RETRIEVER_FRESHNESS_WEIGHT", "0.3")),
    )

    llm_config = LLMConfig(
        groq_api_key = os.getenv("GROQ_API_KEY", ""),
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024")),
        max_chunks = int(os.getenv("LLM_MAX_CHUNKS", "5")),
        max_chunk_len= int(os.getenv("LLM_MAX_CHUNK_LEN", "800")),
    )

    state.retriever = build_retriever(retriever_config)
    state.llm = build_llm_layer(llm_config)

    logger.info("Startup complete ✓")
    yield

    logger.info("Shutting down...")



app = FastAPI(
    title = "Crypto News RAG API",
    description = "Hybrid search + rerank + LLM answer layer over crypto news articles",
    version = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)



class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_n: Optional[int] = Field(default=None, ge=1, le=20)
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    filter: Optional[dict] = Field(
        default=None,
        description="Pinecone metadata filter e.g. {\"tags\": {\"$in\": [\"bitcoin\"]}}"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Where did AlphaTON invest and how much?",
                "top_n": 5,
                "alpha": 0.5,
                "filter": {"tags": {"$in": ["ton", "investment"]}},
            }
        }
    }


class BatchQueryRequest(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=10)
    top_n: Optional[int] = Field(default=None, ge=1, le=20)
    filter: Optional[dict] = None


class QueryResponse(BaseModel):
    question: str
    rag_response: RAGResponse
    retrieved: int          # number of chunks retrieved
    latency_ms: float        # end-to-end latency


class BatchQueryResponse(BaseModel):
    results: list[QueryResponse]
    total: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


class IndexStatsResponse(BaseModel):
    total_vectors: int
    namespaces: dict
    dimension: int



@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
    return response



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again."},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness check",
)
async def health():
    return HealthResponse(status="ok", version=app.version)


@app.get(
    "/index/stats",
    response_model=IndexStatsResponse,
    tags=["System"],
    summary="Pinecone index statistics",
)
async def get_index_stats():
    try:
        stats = index_stats(state.retriever.pinecone_retriever.index)
        return IndexStatsResponse(
            total_vectors=stats.total_vector_count,
            namespaces={k: dict(v) for k, v in stats.namespaces.items()},
            dimension=stats.dimension,
        )
    except Exception as exc:
        logger.exception("Failed to fetch index stats: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not reach Pinecone index.",
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Answer a single question",
)
async def query(req: QueryRequest):
    """
    Full RAG pipeline for a single question:
        Pinecone hybrid search → cross-encoder rerank
        → freshness scoring → LLM answer → structured JSON
    """
    start = time.time()
    logger.info("POST /query | question=%r", req.question[:80])

    # ── Retrieve ─────────────────────────────────────────────────────────
    try:
        chunks: list[ScoredChunk] = retrieve(
            query = req.question,
            retriever = state.retriever,
            top_n = req.top_n,
            alpha = req.alpha,
            filter = req.filter,
        )
    except Exception as exc:
        logger.exception("Retrieval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Retrieval failed: {exc}",
        )

    # ── LLM answer ───────────────────────────────────────────────────────
    try:
        rag_response: RAGResponse = await answer_async(
            question = req.question,
            chunks = chunks,
            llm = state.llm,
        )
    except Exception as exc:
        logger.exception("LLM inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM inference failed: {exc}",
        )

    latency = (time.time() - start) * 1000
    logger.info(
        "POST /query complete | latency=%.0fms | confidence=%.2f | sentiment=%s",
        latency, rag_response.confidence, rag_response.sentiment,
    )

    return QueryResponse(
        question = req.question,
        rag_response = rag_response,
        retrieved = len(chunks),
        latency_ms = round(latency, 1),
    )


@app.post(
    "/query/batch",
    response_model=BatchQueryResponse,
    tags=["RAG"],
    summary="Answer multiple questions in one call",
)
async def query_batch(req: BatchQueryRequest):
    """
    Run the full RAG pipeline for a list of questions.
    Questions are retrieved in batch (shared timestamp for freshness),
    then answered sequentially via the LLM.
    """
    start = time.time()
    logger.info("POST /query/batch | questions=%d", len(req.questions))

    # ── Batch retrieve ───────────────────────────────────────────────────
    try:
        batch_chunks = retrieve_batch(
            queries = req.questions,
            retriever = state.retriever,
            top_n = req.top_n,
            filter = req.filter,
        )
    except Exception as exc:
        logger.exception("Batch retrieval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Batch retrieval failed: {exc}",
        )

    # ── LLM answers ──────────────────────────────────────────────────────
    results = []
    for question in req.questions:
        q_start = time.time()
        chunks  = batch_chunks.get(question, [])

        try:
            rag_response = await answer_async(
                question = question,
                chunks = chunks,
                llm = state.llm,
            )
        except Exception as exc:
            logger.exception("LLM failed for question=%r: %s", question, exc)
            rag_response = RAGResponse(
                answer = "An error occurred while generating this answer.",
                sources = [],
                confidence = 0.0,
                time_sensitivity = "historical",
            )

        results.append(QueryResponse(
            question = question,
            rag_response = rag_response,
            retrieved = len(chunks),
            latency_ms = round((time.time() - q_start) * 1000, 1),
        ))

    total_latency = (time.time() - start) * 1000
    logger.info("POST /query/batch complete | latency=%.0fms", total_latency)

    return BatchQueryResponse(
        results = results,
        total = len(results),
        latency_ms = round(total_latency, 1),
    )