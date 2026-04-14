from __future__ import annotations

import logging
import argparse

from models import ArticleSummary

from client import ArticleApiClient
from config import setup_logging
from repository import ArticleRepository
from runner import ingest_articles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical article backfill.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Maximum number of API pages to backfill. Defaults to all available pages.",
    )
    parser.add_argument(
        "--article-type",
        type=str,
        default="news",
        help="Type of articles (research|news). Defaults to 'research'.",
    )
    return parser.parse_args()

def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger(__name__)
    logger.info("Starting live ingestion with max_pages=%s", args.max_pages)

    with ArticleApiClient() as client, ArticleRepository() as repository:
        summaries = client.iter_pages(article_type=args.article_type, max_pages=args.max_pages)

        # with open('data.json', 'r', encoding='utf-8') as file:
        #     summaries_data = json.load(file)
        #     summaries = [ArticleSummary(**item) for item in summaries_data]
        
        processed, skipped = ingest_articles(
            summaries=summaries,
            repository=repository,
            client=client,
            skip_existing=True,
            article_type=args.article_type,
        )

    logger.info("Live ingestion completed. processed=%s skipped=%s", processed, skipped)
    print(f"Live ingestion completed. processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
