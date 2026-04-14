from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def setup_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(settings.log_level)
        return

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@dataclass(frozen=True)
class Settings:
    base_url: str = os.getenv("ARTICLE_BASE_URL")
    research_base_url: str = os.getenv("RESEARCH_ARTICLE_BASE_URL")
    news_base_url: str = os.getenv("NEWS_ARTICLE_BASE_URL")
    post_base_url: str = os.getenv("ARTICLE_POST_BASE_URL")
    request_timeout: float = float(os.getenv("ARTICLE_REQUEST_TIMEOUT", "30"))
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_database: str = os.getenv("MONGO_DATABASE", "rag_pipeline")
    mongo_collection: str = os.getenv("MONGO_COLLECTION", "articles_raw")
    user_agent: str = os.getenv("ARTICLE_USER_AGENT", "Mozilla/5.0")
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

settings = Settings()


DEFAULT_HEADERS = {
    "User-Agent": settings.user_agent,
    "Accept": "application/json, text/html;q=0.9",
    "Referer": f"{settings.base_url}/research",
}
