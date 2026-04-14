from __future__ import annotations

import logging
from collections.abc import Iterable
import random
import time

import requests

from client import ArticleApiClient
from models import ParsedArticle
from repository import ArticleRepository


def ingest_articles(
    summaries: Iterable,
    repository: ArticleRepository,
    client: ArticleApiClient,
    skip_existing: bool,
    article_type: str,
) -> tuple[int, int]:
    logger = logging.getLogger(__name__)
    inserted_or_updated = 0
    skipped = 0

    for summary in summaries:
        if summary.article_id is None:
            logger.warning("Skipping article with missing ID: title=%s", summary.title)
            skipped += 1
            continue

        if skip_existing and repository.article_exists(summary.article_id):
            logger.info("Skipping existing article %s", summary.article_id)
            skipped += 1
            continue
        
        for attempt in range(10):
            try:
                response = client.fetch_article_content(summary.article_id)
                if response.status_code == 200:
                    article: ParsedArticle = client.parse_article(summary, response.json())
                    article.type = article_type
                    logger.info("Parsed article %s", article.article_id)
                    repository.upsert_article(article)
                    inserted_or_updated += 1
                    time.sleep(0.5)  # rate limit
                    break
                
                if response.status_code == 429:
                    # exponential backoff + jitter
                    wait_time = 1 * (2 ** attempt)
                    jitter = random.uniform(0, 1)
                    sleep_time = wait_time + jitter

                    logger.info(f"429 received. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    response.raise_for_status()

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
    
                wait_time = 1 * (2 ** attempt)
                time.sleep(wait_time)

    logger.info(
        "Ingestion finished. processed=%s skipped=%s",
        inserted_or_updated,
        skipped,
    )
    return inserted_or_updated, skipped
