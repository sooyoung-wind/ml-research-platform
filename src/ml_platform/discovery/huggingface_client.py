"""ML Research Platform — HuggingFace Papers API client.

Async client for searching and fetching trending papers from
HuggingFace's daily papers endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import types
from datetime import date, datetime, timezone
from typing import Any

import httpx

from ml_platform.config import APIConfig
from ml_platform.models import Author, Paper, PaperSource

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Async client for the HuggingFace Papers API.

    Attributes:
        _base_url: HuggingFace Papers base URL.
        _rate_limit: Minimum interval between requests in seconds.
        _timeout: HTTP request timeout in seconds.
        _client: Underlying httpx async client (lazily created).
        _last_request_time: Timestamp of the last request for rate limiting.

    Usage::

        async with HuggingFaceClient() as client:
            trending = await client.get_trending(limit=20)
    """

    def __init__(
        self,
        base_url: str | None = None,
        rate_limit: float | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the HuggingFaceClient.

        Args:
            base_url: HuggingFace base URL.
            rate_limit: Requests per second.
            timeout: HTTP request timeout in seconds.
        """
        self._base_url = base_url or APIConfig.HUGGINGFACE_PAPERS_BASE_URL
        self._rate_limit = rate_limit if rate_limit is not None else APIConfig.HUGGINGFACE_PAPERS_RATE_LIMIT
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def __aenter__(self) -> HuggingFaceClient:
        """Enter the async context manager and create the HTTP client.

        Returns:
            The HuggingFaceClient instance.
        """
        self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the async context manager and close the HTTP client.

        Args:
            exc_type: Exception type, or None if no exception occurred.
            exc_val: Exception instance, or None.
            exc_tb: Traceback object, or None.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the active httpx client, creating a lazy one if needed.

        Returns:
            The active httpx.AsyncClient instance.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self._client

    async def _rate_limit_sleep(self) -> None:
        """Sleep to respect the configured rate limit.

        This ensures a minimum delay between consecutive HTTP requests
        based on the configured ``_rate_limit`` value.
        """
        import time

        now = time.monotonic()
        elapsed = now - self._last_request_time
        wait = self._rate_limit - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_time = time.monotonic()

    # --- Public API ---------------------------------------------------------

    async def get_trending(self, limit: int = 20) -> list[Paper]:
        """Fetch today's trending papers from HuggingFace.

        Args:
            limit: Maximum number of papers to return.

        Returns:
            A list of :class:`Paper` objects sorted by upvotes
            (descending).

        Raises:
            httpx.HTTPStatusError: On non-2xx HTTP responses (logged, not re-raised).
            httpx.RequestError: On network errors (logged, not re-raised).
        """
        await self._rate_limit_sleep()

        url = f"{self._base_url}/api/daily_papers"
        params = {"date": date.today().isoformat()}

        try:
            client = self._get_client()
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("HuggingFace HTTP error: %s", exc)
            return []
        except httpx.RequestError as exc:
            logger.error("HuggingFace network error: %s", exc)
            return []

        # Sort by upvotes descending
        data.sort(key=lambda p: p.get("paper", {}).get("upvotes", 0), reverse=True)

        papers: list[Paper] = []
        for item in data[:limit]:
            paper = self._parse_paper(item)
            if paper is not None:
                papers.append(paper)

        return papers

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Paper]:
        """Search HuggingFace papers by keyword.

        Note: HuggingFace does not have a full-text search API for papers.
        This fetches recent daily papers and filters client-side by title.

        Args:
            query: Search query string.
            limit: Maximum number of papers to return.

        Returns:
            A list of matching :class:`Paper` objects.

        Raises:
            httpx.HTTPStatusError: On non-2xx HTTP responses (logged, not re-raised).
            httpx.RequestError: On network errors (logged, not re-raised).
        """
        all_papers = await self._fetch_recent_papers()
        matched = self._filter_papers_by_query(all_papers, query)
        return self._dedupe_and_sort(matched, limit)

    async def _fetch_recent_papers(self, days: int = 7) -> list[Paper]:
        """Fetch papers from the last several days.

        Args:
            days: Number of days to look back.

        Returns:
            A list of all papers found in the date range.
        """
        all_papers: list[Paper] = []
        today = date.today()

        for i in range(days):
            d = today - __import__("datetime").timedelta(days=i)
            await self._rate_limit_sleep()

            url = f"{self._base_url}/api/daily_papers"
            params = {"date": d.isoformat()}

            try:
                client = self._get_client()
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning("HuggingFace fetch failed for %s: %s", d, exc)
                continue

            for item in data:
                paper = self._parse_paper(item)
                if paper is not None:
                    all_papers.append(paper)

        return all_papers

    @staticmethod
    def _filter_papers_by_query(papers: list[Paper], query: str) -> list[Paper]:
        """Filter papers by a case-insensitive query match in title/abstract.

        Args:
            papers: List of papers to filter.
            query: Search query string.

        Returns:
            A list of papers matching the query.
        """
        q_lower = query.lower()
        return [
            p for p in papers
            if q_lower in (p.title or "").lower()
            or q_lower in (p.abstract or "").lower()
        ]

    @staticmethod
    def _dedupe_and_sort(papers: list[Paper], limit: int) -> list[Paper]:
        """Deduplicate papers by arxiv_id and sort by upvotes descending.

        Args:
            papers: List of papers to deduplicate and sort.
            limit: Maximum number of papers to return.

        Returns:
            A deduplicated, sorted, and truncated list of papers.
        """
        seen: set[str] = set()
        unique: list[Paper] = []
        for p in papers:
            if p.arxiv_id and p.arxiv_id not in seen:
                seen.add(p.arxiv_id)
                unique.append(p)
        unique.sort(key=lambda p: p.upvotes, reverse=True)
        return unique[:limit]

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        """Fetch a single paper by its arXiv ID.

        Args:
            arxiv_id: An arXiv identifier.

        Returns:
            A :class:`Paper` object, or ``None`` if not found.

        Raises:
            httpx.HTTPStatusError: On non-2xx HTTP responses (logged, not re-raised).
            httpx.RequestError: On network errors (logged, not re-raised).
        """
        url = f"{self._base_url}/api/daily_papers/{arxiv_id}"

        try:
            client = self._get_client()
            resp = await client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("HuggingFace HTTP error: %s", exc)
            return None
        except httpx.RequestError as exc:
            logger.error("HuggingFace network error: %s", exc)
            return None

        return self._parse_paper(data)

    # --- Internal -----------------------------------------------------------

    def _parse_paper(self, item: dict[str, Any]) -> Paper | None:
        """Parse a HuggingFace paper dict into a Paper model.

        Returns None if the entry cannot be parsed.

        Args:
            item: A dict from the HuggingFace daily papers API.

        Returns:
            A Paper object, or None on failure.
        """
        try:
            paper_data = item.get("paper", item)
            arxiv_id = paper_data.get("id", "")
            title = self._resolve_title(paper_data, item)
            if not title:
                return None

            authors = self._parse_authors(paper_data)
            published_date = self._parse_published_date(paper_data)
            summary = paper_data.get("summary") or item.get("abstract")
            upvotes = paper_data.get("upvotes", 0)
            code_url = self._extract_code_url(paper_data)

            return Paper(
                paper_id=arxiv_id,
                source=PaperSource.HUGGINGFACE,
                title=title,
                abstract=summary,
                authors=authors,
                published_date=published_date,
                categories=paper_data.get("categories", []),
                arxiv_id=arxiv_id,
                url=f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else None,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
                code_url=code_url,
                upvotes=upvotes,
                year=published_date.year if published_date else None,
            )
        except Exception:
            logger.exception("Failed to parse HuggingFace paper entry")
            return None

    @staticmethod
    def _resolve_title(paper_data: dict[str, Any], item: dict[str, Any]) -> str:
        """Resolve the paper title from nested data structures.

        Args:
            paper_data: The nested paper data dict.
            item: The top-level item dict.

        Returns:
            The resolved title string, or empty string if not found.
        """
        title = paper_data.get("title", "")
        if not title:
            title = item.get("title", "")
        return title

    @staticmethod
    def _parse_authors(paper_data: dict[str, Any]) -> list[Author]:
        """Parse the authors list from HuggingFace paper data.

        Args:
            paper_data: The paper data dict containing authors.

        Returns:
            A list of Author objects.
        """
        authors_list = paper_data.get("authors", [])
        return [Author(name=a.get("name", str(a))) for a in authors_list if a]

    @staticmethod
    def _parse_published_date(paper_data: dict[str, Any]) -> datetime | None:
        """Parse the published date from paper data.

        Args:
            paper_data: The paper data dict with a published date string.

        Returns:
            A datetime object, or None on parse failure.
        """
        published_str = paper_data.get("published")
        if not published_str:
            return None
        try:
            return datetime.fromisoformat(
                published_str.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_code_url(paper_data: dict[str, Any]) -> str | None:
        """Extract the first GitHub URL from paper data.

        Args:
            paper_data: The paper data dict that may contain github_urls.

        Returns:
            The first GitHub URL, or None if not found.
        """
        github_urls = paper_data.get("github_urls", [])
        return github_urls[0] if github_urls else None
