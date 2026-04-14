"""
Chainlit chatbot for the crypto news RAG pipeline.

Wires directly to:
    retrieval.retriever  → PineconeHybridSearchRetriever + cross-encoder + freshness
    llm_client           → ChatGroq + structured RAGResponse

Run:
    chainlit run app.py
"""

import logging
import os
import chainlit as cl
from dotenv import load_dotenv

from langsmith import Client
from langchain_core.tracers.context import collect_runs
from src.retrieval.retriever import RetrieverConfig, build_retriever, retrieve
from llm_client import LLMConfig, LLMLayer, build_llm_layer, answer_async

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


retriever = build_retriever(
    RetrieverConfig(
        pinecone_api_key = os.getenv("PINECONE_API_KEY", ""),
        pinecone_index = os.getenv("PINECONE_INDEX", "crypto-news"),
        pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region = os.getenv("PINECONE_REGION", "us-east-1"),
        bm25_path = os.getenv("BM25_PATH", "bm25_encoder.json"),
        top_k = int(os.getenv("RETRIEVER_TOP_K", "20")),
        rerank_top_n = int(os.getenv("RETRIEVER_TOP_N", "5")),
        alpha = float(os.getenv("RETRIEVER_ALPHA", "0.5")),
        lambda_ = float(os.getenv("RETRIEVER_LAMBDA", "0.1")),
        freshness_weight = float(os.getenv("RETRIEVER_FRESHNESS_WEIGHT", "0.3")),
    )
)

llm: LLMLayer = build_llm_layer(
    LLMConfig(
        groq_api_key = os.getenv("GROQ_API_KEY", ""),
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_chunks = int(os.getenv("LLM_MAX_CHUNKS", "5")),
        max_chunk_len = int(os.getenv("LLM_MAX_CHUNK_LEN", "800")),
    )
)

logger.info("Retriever and LLM layer ready ✓")



@cl.set_starters
async def set_starters():
    return [
        cl.Starter(          # ← capital S
            label = "Latest BTC news",
            message = "What is the latest news about Bitcoin?",
        ),
        cl.Starter(          # ← capital S
            label = "ETH market update",
            message = "What is happening with Ethereum right now?",
        ),
        cl.Starter(          # ← capital S
            label = "Top movers today",
            message = "Which crypto assets are making big moves today?",
        ),
        cl.Starter(          # ← capital S
            label = "Market sentiment",
            message = "What is the overall crypto market sentiment this week?",
        ),
    ]



# @cl.on_chat_start
# async def on_chat_start():
#     cl.user_session.set("history", [])
#     await cl.Message(
#         content=(
#             "Hi! I'm your crypto market assistant. "
#             "Ask me anything about the latest crypto news, "
#             "price movements, or market trends."
#         ),
#         author="assistant",
#     ).send()



@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()

    if not question:
        return

    # Show a thinking indicator while retrieving + generating
    async with cl.Step(name="Searching articles", show_input=False) as step:
        try:
            chunks = retrieve(question, retriever, top_n=5)
            step.output = f"Found {len(chunks)} relevant articles"
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            step.output = "Retrieval failed"
            await cl.Message(
                content="Sorry, I had trouble searching the articles. Please try again.",
            ).send()
            return

    # LLM answer
    async with cl.Step(name="Generating answer", show_input=False) as step:
        try:
            response = await answer_async(question, chunks, llm)
            step.output = f"Confidence: {response.confidence:.0%}"
        except Exception as exc:
            logger.exception("LLM inference failed: %s", exc)
            await cl.Message(
                content="Sorry, something went wrong generating the answer. Please try again.",
            ).send()
            return

    # ── Main answer ───────────────────────────────────────────────────────
    answer_text = response.answer

    # Append confidence badge inline
    confidence_pct = int(response.confidence * 100)
    if confidence_pct >= 75:
        badge = f"🟢 {confidence_pct}% confidence"
    elif confidence_pct >= 50:
        badge = f"🟡 {confidence_pct}% confidence"
    else:
        badge = f"🔴 {confidence_pct}% confidence — limited sources"

    await cl.Message(
        content=f"{answer_text}\n\n*{badge}*",
        author="assistant",
    ).send()

    # ── Source citations ──────────────────────────────────────────────────
    # if response.sources:
    #     source_elements = []

    #     for src in response.sources:
    #         date    = src.published_at[:10] if src.published_at else "unknown"
    #         content = (
    #             f"**{src.title}**\n"
    #             f"Published: {date}\n"
    #             f"[Read article]({src.url})\n\n"
    #             f"{src.snippet}"
    #         )
    #         source_elements.append(
    #             cl.Text(
    #                 name = src.title[:40] + "..." if len(src.title) > 40 else src.title,
    #                 content = content,
    #                 display = "side",
    #             )
    #         )

    #     await cl.Message(
    #         content = f"**Sources** ({len(response.sources)} articles)",
    #         elements = source_elements,
    #         author = "assistant",
    #     ).send()

    # ── Follow-up suggestion ──────────────────────────────────────────────
    if response.follow_up:
        actions = [
            cl.Action(
                name = "follow_up",
                label = f"➜  {response.follow_up}",
                payload = {"question": response.follow_up},
            )
        ]
        await cl.Message(
            content = "You might also want to ask:",
            actions = actions,
            author = "assistant",
        ).send()



@cl.action_callback("follow_up")
async def on_follow_up(action: cl.Action):
    """Re-run the pipeline when user clicks a follow-up suggestion."""
    question = action.payload.get("question", "")
    if question:
        # Simulate the user sending the follow-up as a new message
        await on_message(cl.Message(content=question))