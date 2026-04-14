from __future__ import annotations

import logging

from pymongo.mongo_client import MongoClient, UpdateOne
from pymongo.server_api import ServerApi
from pymongo.collection import Collection

from config import settings
from models import ParsedArticle


class ArticleRepository:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._client = MongoClient(settings.mongo_uri, server_api=ServerApi('1'))
        self._collection: Collection = self._client[settings.mongo_database][
            settings.mongo_collection
        ]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self._collection.create_index("article_id", unique=True)
        self._collection.create_index("slug")
        self._collection.create_index("published_at")
        self._logger.debug("MongoDB indexes ensured for collection %s", settings.mongo_collection)

    def close(self) -> None:
        self._logger.debug("Closing MongoDB client")
        self._client.close()

    def article_exists(self, article_id: int | None) -> bool:
        if article_id is None:
            return False
        exists = self._collection.count_documents({"article_id": article_id}, limit=1) > 0
        if exists:
            self._logger.debug("Article %s already exists in MongoDB", article_id)
        return exists

    def upsert_article(self, article: ParsedArticle) -> None:
        if article.article_id is None:
            raise ValueError("Article ID is required for upsert operations.")

        document = article.to_document()
        self._collection.update_one(
            {"article_id": article.article_id},
            {"$set": document},
            upsert=True,
        )
        self._logger.info("Upserted article %s", article.article_id)

    def bulk_upsert(self, articles: list[ParsedArticle]) -> int:
        operations = []
        for article in articles:
            if article.article_id is None:
                continue
            operations.append(
                UpdateOne(
                    {"article_id": article.article_id},
                    {"$set": article.to_document()},
                    upsert=True,
                )
            )

        if not operations:
            return 0

        result = self._collection.bulk_write(operations, ordered=False)
        self._logger.info(
            "Bulk upsert completed. upserted=%s modified=%s",
            result.upserted_count,
            result.modified_count,
        )
        return result.upserted_count + result.modified_count

    def __enter__(self) -> "ArticleRepository":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
