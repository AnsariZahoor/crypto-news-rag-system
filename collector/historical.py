from __future__ import annotations

import argparse
import logging

from client import ArticleApiClient
from repository import ArticleRepository
from runner import ingest_articles

from config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical article backfill.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3326,
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
    logger.info("Starting historical ingestion with max_pages=%s", args.max_pages)

    with ArticleApiClient() as client, ArticleRepository() as repository:
        processed = 0
        skipped = 0

        for page_number, summaries in enumerate(
            client.iter_page_batches(
                max_pages=args.max_pages,
                article_type=args.article_type,
            ),
            start=1,
        ):
            logger.info(
                "Processing historical page %s with %s summaries",
                page_number,
                len(summaries),
            )
            page_processed, page_skipped = ingest_articles(
                summaries=summaries,
                repository=repository,
                client=client,
                skip_existing=False,
                article_type=args.article_type,
            )
            processed += page_processed
            skipped += page_skipped

    logger.info("Historical ingestion completed. processed=%s skipped=%s", processed, skipped)
    print(f"Historical ingestion completed. processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
