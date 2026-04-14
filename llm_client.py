"""
Flow:
    ScoredChunk list (from retriever)
        → format_context()
        → ChatPromptTemplate
        → ChatGroq
        → with_structured_output()
        → RAGResponse
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.retrieval.retriever import RetrieverConfig, build_retriever, retrieve

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_chunks: int = 5
    max_chunk_len: int = 800



class SourceArticle(BaseModel):
    article_id: str
    title: str
    url: str
    published_at: str
    snippet: str = Field(description="Excerpt that directly supports the answer")


class RAGResponse(BaseModel):
    answer: str = Field(description="Factual answer grounded only in the articles")
    sources: list[SourceArticle]
    confidence: float = Field(ge=0.0, le=1.0)
    follow_up: Optional[str] = Field(
        default=None,
        description="A suggested follow-up question the user might want to ask next"
    )


SYSTEM_TEMPLATE = """\
You are a helpful crypto market assistant. Answer questions using ONLY the articles below.

Rules:
- Ground every claim in the provided articles.
- Cite sources using article_id.
- If the articles do not contain enough information, say so honestly and set confidence low.
- Keep the answer concise and conversational — this is a chatbot interface.
- Suggest a relevant follow-up question the user might find useful.

Current UTC time: {current_datetime}

--- ARTICLES ---
{context}
--- END ARTICLES ---
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human",  "{question}"),
])



@dataclass
class LLMLayer:
    config: LLMConfig
    chain: Any


def build_llm_layer(config: LLMConfig) -> LLMLayer:
    llm = ChatGroq(
        api_key=config.groq_api_key,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    chain = _PROMPT | llm.with_structured_output(RAGResponse)
    logger.info("LLM layer built | model=%s", config.model)
    return LLMLayer(config=config, chain=chain)



def format_context(
    chunks: list[Any],
    max_chunk_len: int = 800,
) -> str:
    parts = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            meta = chunk
            content = chunk.get("content", "")
            score = chunk.get("score", 0.0)
        else:
            meta = chunk.metadata
            content = chunk.content
            score = chunk.final_score

        if len(content) > max_chunk_len:
            content = content[:max_chunk_len] + "..."

        parts.append(
            f"[article_id: {meta.get('article_id', '')}]\n"
            f"Title: {meta.get('title', '')}\n"
            f"Published: {meta.get('published_at', '')}\n"
            f"Score: {score:.4f}\n\n"
            f"{content}"
        )

    return ("\n" + "=" * 60 + "\n").join(parts)



def answer(
    question: str,
    chunks: list[Any],
    llm: LLMLayer,
) -> RAGResponse:
    config  = llm.config
    trimmed = chunks[: config.max_chunks]

    if not trimmed:
        return RAGResponse(
            answer="I couldn't find any relevant articles to answer that question.",
            sources=[],
            confidence=0.0,
            follow_up="Could you rephrase or ask about a specific coin or event?",
        )

    try:
        response: RAGResponse = llm.chain.invoke({
            "context": format_context(trimmed, config.max_chunk_len),
            "question": question,
            "current_datetime": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        })
        logger.info(
            "answer | query=%r | confidence=%.2f | sources=%d",
            question[:60], response.confidence, len(response.sources),
        )
        return response

    except Exception as exc:
        logger.exception("LLM chain failed: %s", exc)
        return RAGResponse(
            answer="Something went wrong. Please try again.",
            sources=[],
            confidence=0.0,
        )


@traceable(
    name="crypto-rag-answer",
    tags=["rag", "crypto"],
)
async def answer_async(
    question: str,
    chunks: list[Any],
    llm: LLMLayer,
) -> RAGResponse:
    config = llm.config
    trimmed = chunks[: config.max_chunks]

    if not trimmed:
        return RAGResponse(
            answer="I couldn't find any relevant articles to answer that question.",
            sources=[],
            confidence=0.0,
            follow_up="Could you rephrase or ask about a specific coin or event?",
        )

    try:
        response: RAGResponse = await llm.chain.ainvoke({
            "context": format_context(trimmed, config.max_chunk_len),
            "question": question,
            "current_datetime": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        })
        logger.info(
            "answer_async | query=%r | confidence=%.2f | sources=%d",
            question[:60], response.confidence, len(response.sources),
        )
        return response

    except Exception as exc:
        logger.exception("Async LLM chain failed: %s", exc)
        return RAGResponse(
            answer="Something went wrong. Please try again.",
            sources=[],
            confidence=0.0,
        )



if __name__ == "__main__":
    import json

    query = "Where does Bitcoin stand right now and what are the major news affecting it?"
    retriever = build_retriever(RetrieverConfig())
    llm = build_llm_layer(LLMConfig())

    chunks   = retrieve(query, retriever, top_n=5)
    response = answer(query, chunks, llm)

    print(response.answer)
    print(f"\nConfidence: {response.confidence:.0%}")
    print(f"Follow-up: {response.follow_up}")
    print("\nSources:")
    for src in response.sources:
        print(f" [{src.article_id}] {src.published_at[:10]} — {src.title}")
        print(f" {src.snippet[:100]}")

    print("\nPayload:")
    print(json.dumps(response.model_dump(), indent=2))