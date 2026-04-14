from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

from gridfs.grid_file import Optional
import httpx

from config import DEFAULT_HEADERS, settings
from models import ArticleSummary, ParsedArticle


PAGE_FETCH_MAX_RETRIES = 5
PAGE_FETCH_BASE_DELAY_SECONDS = 1.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / ".ingestion_page_checkpoint.json"


class ArticleApiClient:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._client = httpx.Client(
            headers=DEFAULT_HEADERS,
            timeout=settings.request_timeout,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._logger.debug("Closing HTTP client")
        self._client.close()

    def fetch_article_page(self, start: int = 0, article_type: str = "research") -> dict:
        self._logger.info("Fetching article page", extra={"start": start, "type": article_type})
        base_url = getattr(settings, f"{article_type}_base_url", settings.research_base_url)
        response = self._client.get(
            base_url,
            params={"start": start},
        )
        response.raise_for_status()
        return response.json()

    def _load_page_checkpoint(self, article_type: str) -> int:
        if not CHECKPOINT_PATH.exists():
            return 0

        try:
            payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.warning("Failed to read page checkpoint: %s", exc)
            return 0

        checkpoint = payload.get(article_type, {})
        next_start = checkpoint.get("next_start", 0)
        return next_start if isinstance(next_start, int) and next_start >= 0 else 0

    def _save_page_checkpoint(
        self,
        article_type: str,
        next_start: int,
        pages_fetched: int,
        last_batch_size: int,
    ) -> None:
        payload: dict[str, dict] = {}
        if CHECKPOINT_PATH.exists():
            try:
                payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._logger.warning("Failed to parse existing checkpoint, overwriting it: %s", exc)

        payload[article_type] = {
            "next_start": next_start,
            "pages_fetched": pages_fetched,
            "last_batch_size": last_batch_size,
            "updated_at": int(time.time()),
        }

        CHECKPOINT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _clear_page_checkpoint(self, article_type: str) -> None:
        if not CHECKPOINT_PATH.exists():
            return

        try:
            payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.warning("Failed to parse checkpoint during cleanup: %s", exc)
            return

        if article_type not in payload:
            return

        del payload[article_type]

        if payload:
            CHECKPOINT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            CHECKPOINT_PATH.unlink(missing_ok=True)

    def _fetch_article_page_with_retry(self, start: int, article_type: str) -> dict:
        for attempt in range(PAGE_FETCH_MAX_RETRIES):
            try:
                return self.fetch_article_page(start=start, article_type=article_type)
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code not in RETRYABLE_STATUS_CODES or attempt == PAGE_FETCH_MAX_RETRIES - 1:
                    raise

                retry_after = exc.response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_seconds = float(retry_after)
                else:
                    sleep_seconds = PAGE_FETCH_BASE_DELAY_SECONDS * (2**attempt)

                self._logger.warning(
                    "Page fetch failed with status %s for start=%s type=%s. Retrying in %.1fs.",
                    status_code,
                    start,
                    article_type,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
            except httpx.RequestError as exc:
                if attempt == PAGE_FETCH_MAX_RETRIES - 1:
                    raise

                sleep_seconds = PAGE_FETCH_BASE_DELAY_SECONDS * (2**attempt)
                self._logger.warning(
                    "Page fetch request error for start=%s type=%s: %s. Retrying in %.1fs.",
                    start,
                    article_type,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

    def fetch_article_content(self, article_id: int) -> dict:
        self._logger.info(f"Fetching article id", extra={"article_id": article_id})
        url = f"{settings.post_base_url}/{article_id}"
        response = self._client.get(url)
        return response
        # response.raise_for_status()
        # return response.json()

    def extract_summaries(self, payload: dict) -> list[ArticleSummary]:
        posts = (
            payload.get("data", {})
            .get("content", {})
            .get("posts", [])
        )

        articles: list[ArticleSummary] = []
        for item in posts:
            article_id=item.get("id")
            articles.append(
                ArticleSummary(
                    article_id=article_id,
                    title=item.get("title"),
                    slug=item.get("slug")
                )
            )

        self._logger.info("Extracted %s article summaries from payload", len(articles))
        return articles

    def iter_page_batches(
        self, article_type: str, max_pages: int | None = None
    ) -> Iterator[list[ArticleSummary]]:
        start = self._load_page_checkpoint(article_type)
        pages_fetched = 0
        self._logger.info(
            "Starting paginated iteration for type=%s from start=%s",
            article_type,
            start,
        )

        while True:
            if max_pages is not None and pages_fetched >= max_pages:
                break

            payload = self._fetch_article_page_with_retry(start=start, article_type=article_type)
            articles = self.extract_summaries(payload)
            if not articles:
                self._logger.info("No more articles returned from API, stopping pagination")
                self._clear_page_checkpoint(article_type)
                break

            pages_fetched += 1
            page_size = len(articles)
            start += page_size
            self._save_page_checkpoint(
                article_type=article_type,
                next_start=start,
                pages_fetched=pages_fetched,
                last_batch_size=page_size,
            )
            yield articles
            time.sleep(1)  # Sleep to avoid overwhelming the API

        self._logger.info("Pagination completed across %s pages", pages_fetched)

    def iter_pages(self, article_type: str, max_pages: int | None = None) -> list[ArticleSummary]:
        all_articles: list[ArticleSummary] = []
        pages_fetched = 0
        for articles in self.iter_page_batches(article_type=article_type, max_pages=max_pages):
            all_articles.extend(articles)
            pages_fetched += 1

        self._logger.info(
            "Pagination completed with %s summaries across %s pages",
            len(all_articles),
            pages_fetched,
        )
        return all_articles

    def parse_article(self, summary: ArticleSummary, payload: dict) -> Optional[ParsedArticle]:
        if not payload or not payload.get("success"):
            return None

        text = payload.get("data") or {}
        # # write to json file for debugging --- IGNORE ---
        # with open(f"article_{summary.article_id}.json", "w") as f:
        #     json.dump(text, f, indent=2)
        meta = text.get("meta") or {}
        script = meta.get("script")[0] if meta.get("script") and isinstance(meta.get("script"), list) else {}

        tags = [
            tag["slug"]
            for tag in text.get("tags", [])
            if isinstance(tag, dict) and tag.get("slug")
        ]

        content = script.get('json').get("articleBody") or ""
        published_at = script.get('json').get("datePublished") or ""

        return ParsedArticle(
            article_id=summary.article_id,
            title=summary.title,
            content=content,
            url=text.get("url") or "",
            slug=summary.slug,
            tags=tags,
            thumbnail=text.get("thumbnail") or "",
            published_at=published_at
        )
    
    def __enter__(self) -> "ArticleApiClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
