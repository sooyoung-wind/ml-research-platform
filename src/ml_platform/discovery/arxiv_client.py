"""ML Research Platform — arXiv API client.

Async client for searching and fetching papers from the arXiv API.
Uses httpx for HTTP requests and defusedxml for safe XML parsing.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from defusedxml import ElementTree as SafeET

from ml_platform.config import APIConfig
from ml_platform.models import Author, Paper, PaperSource

logger = logging.getLogger(__name__)

# XML namespace map used by the arXiv Atom feed
_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


def _ns(tag: str, prefix: str = "atom") -> str:
    """Return a fully-qualified element tag for the given namespace prefix."""
    return f"{{{_NAMESPACES[prefix]}}}{tag}"


def _strip_arxiv_version(arxiv_id: str) -> str:
    """Remove the version suffix (e.g. 'v1') from an arXiv ID.

    >>> _strip_arxiv_version('2301.07041v1')
    '2301.07041'
    >>> _strip_arxiv_version('2301.07041')
    '2301.07041'
    """
    return re.sub(r"v\d+$", "", arxiv_id)


def _parse_entry(entry: Any) -> Paper | None:
    """Parse a single <entry> XML element into a Paper model.

    Returns None if the entry cannot be parsed (e.g. missing required fields).
    """
    try:
        # --- arXiv ID ---
        id_text: str | None = entry.findtext(_ns("id"))
        if id_text is None:
            logger.warning("Skipping entry with no <id>")
            return None
        # id is a URL like https://arxiv.org/abs/2301.07041v1
        raw_id = id_text.rstrip("/").split("/")[-1]
        clean_id = _strip_arxiv_version(raw_id)

        # --- Title ---
        title = (entry.findtext(_ns("title")) or "").strip().replace("\n", " ")
        # Normalise whitespace
        title = re.sub(r"\s+", " ", title)

        # --- Abstract ---
        summary = (entry.findtext(_ns("summary")) or "").strip().replace("\n", " ")
        summary = re.sub(r"\s+", " ", summary) or None

        # --- Authors ---
        authors: list[Author] = []
        for author_el in entry.findall(_ns("author")):
            name = author_el.findtext(_ns("name"))
            if name:
                authors.append(Author(name=name.strip()))

        # --- Published date ---
        published_str = entry.findtext(_ns("published"))
        published_date: datetime | None = None
        if published_str:
            published_date = _parse_datetime(published_str)

        # --- Categories ---
        categories: list[str] = []
        # Primary category
        primary_cat = entry.find(_ns("primary_category", "arxiv"))
        if primary_cat is not None:
            term = primary_cat.get("term")
            if term:
                categories.append(term)
        # Additional categories from <category> elements
        for cat_el in entry.findall(_ns("category")):
            term = cat_el.get("term")
            if term and term not in categories:
                categories.append(term)

        # --- PDF URL ---
        pdf_url = f"https://arxiv.org/pdf/{clean_id}"

        # --- DOI (optional) ---
        doi_el = entry.find(_ns("doi", "arxiv"))
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        # --- Comment / venue hint (optional) ---
        # arXiv entries sometimes have a <arxiv:comment> with conference info

        return Paper(
            paper_id=clean_id,
            source=PaperSource.ARXIV,
            doi=doi,
            title=title,
            abstract=summary,
            authors=authors,
            published_date=published_date,
            categories=categories,
            arxiv_id=clean_id,
            url=f"https://arxiv.org/abs/{clean_id}",
            pdf_url=pdf_url,
            year=published_date.year if published_date else None,
        )
    except Exception:
        logger.exception("Failed to parse arXiv entry")
        return None


def _parse_datetime(dt_str: str) -> datetime | None:
    """Parse an ISO 8601 datetime string, returning None on failure."""
    try:
        # arXiv returns e.g. '2023-01-17T18:43:42Z'
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class ArxivClient:
    """Async client for the arXiv API.

    Usage::

        async with ArxivClient() as client:
            papers = await client.search_by_keyword("diffusion models")
    """

    def __init__(
        self,
        base_url: str | None = None,
        rate_limit: float | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url or APIConfig.ARXIV_BASE_URL
        self._rate_limit = rate_limit if rate_limit is not None else APIConfig.ARXIV_RATE_LIMIT
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    # --- Context manager ---------------------------------------------------

    async def __aenter__(self) -> "ArxivClient":
        self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- Internal helpers ---------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return the active httpx client, creating a lazy one if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self._client

    async def _rate_limit_sleep(self) -> None:
        """Sleep to respect the configured rate limit."""
        import time

        now = time.monotonic()
        elapsed = now - self._last_request_time
        wait = self._rate_limit - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_time = time.monotonic()

    async def _fetch(self, params: dict[str, Any]) -> str:
        """Execute a rate-limited GET request and return the response body."""
        await self._rate_limit_sleep()

        url = f"{self._base_url}?{urlencode(params)}"
        logger.debug("arXiv request: %s", url)

        client = self._get_client()
        response = await client.get(url)
        response.raise_for_status()
        return response.text

    def _parse_feed(self, xml_body: str) -> list[Paper]:
        """Parse the Atom XML response body and return a list of Paper objects."""
        try:
            root = SafeET.fromstring(xml_body)
        except Exception:
            logger.exception("Failed to parse arXiv XML response")
            return []

        entries = root.findall(_ns("entry"))
        if not entries:
            logger.debug("No entries found in arXiv response")
            return []

        papers: list[Paper] = []
        for entry in entries:
            paper = _parse_entry(entry)
            if paper is not None:
                papers.append(paper)

        return papers

    # --- Public API ---------------------------------------------------------

    async def search_by_keyword(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[Paper]:
        """Search arXiv by keyword query.

        Args:
            query: Free-text search query (e.g. ``"diffusion models"``).
            max_results: Maximum number of results to return.

        Returns:
            A list of :class:`Paper` objects.
        """
        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            xml_body = await self._fetch(params)
            return self._parse_feed(xml_body)
        except httpx.HTTPStatusError as exc:
            logger.error("arXiv HTTP error: %s", exc)
            return []
        except httpx.RequestError as exc:
            logger.error("arXiv network error: %s", exc)
            return []

    async def search_by_category(
        self,
        category: str = "cs.AI",
        max_results: int = 10,
    ) -> list[Paper]:
        """Search arXiv by category.

        Args:
            category: An arXiv category (e.g. ``"cs.AI"``, ``"cs.LG"``).
            max_results: Maximum number of results to return.

        Returns:
            A list of :class:`Paper` objects sorted by submission date (newest first).
        """
        params = {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            xml_body = await self._fetch(params)
            return self._parse_feed(xml_body)
        except httpx.HTTPStatusError as exc:
            logger.error("arXiv HTTP error: %s", exc)
            return []
        except httpx.RequestError as exc:
            logger.error("arXiv network error: %s", exc)
            return []

    async def get_recent(
        self,
        categories: list[str] | None = None,
        days: int = 7,
        max_results: int = 50,
    ) -> list[Paper]:
        """Fetch recent papers from one or more categories.

        Builds a query for papers submitted within the last *days* days across
        the given categories, sorted by submission date (newest first).

        Args:
            categories: List of arXiv categories (defaults to
                ``["cs.AI", "cs.LG", "cs.CV"]``).
            days: How many days back to look.
            max_results: Maximum number of results to return.

        Returns:
            A list of :class:`Paper` objects.
        """
        if categories is None:
            categories = ["cs.AI", "cs.LG", "cs.CV"]

        # Build a category OR query: cat:cs.AI OR cat:cs.LG OR ...
        cat_query = " OR ".join(f"cat:{c}" for c in categories)

        # Build a date filter: submittedDate >= (now - days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        # arXiv date-range filter format: submittedDate:[YYYYMMDDTTTT TO YYYYMMDDTTTT]
        cutoff_str = cutoff.strftime("%Y%m%d%H%M")
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        date_query = f"submittedDate:[{cutoff_str} TO {now_str}]"

        search_query = f"({cat_query}) AND {date_query}"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            xml_body = await self._fetch(params)
            return self._parse_feed(xml_body)
        except httpx.HTTPStatusError as exc:
            logger.error("arXiv HTTP error: %s", exc)
            return []
        except httpx.RequestError as exc:
            logger.error("arXiv network error: %s", exc)
            return []
