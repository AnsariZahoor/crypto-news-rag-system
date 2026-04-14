from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ArticleSummary:
    article_id: int | None
    title: str | None
    slug: str | None


@dataclass
class ParsedArticle:
    article_id: int | None
    title: str | None
    content: str | None
    url: str | None
    slug: str | None
    tags: list[str]
    thumbnail: str | None
    published_at: str | None
    type: str = None
    source: str = "theblock"
    def to_document(self) -> dict[str, Any]:
        return asdict(self)
