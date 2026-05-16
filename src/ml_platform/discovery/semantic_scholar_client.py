"""ML Research Platform — Semantic Scholar Graph API v1 client.

Async client for paper search, retrieval, citation lookup, and
metadata enrichment via the Semantic Scholar API.
"""

from __future__ import annotations

import asyncio
import logging
import types
from typing import Any

import httpx

from ml_platform.config import APIConfig
from ml_platform.models import Author, Paper, PaperSource

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_FIELDS = (
    "title,abstract,authors,year,citationCount,isOpenAccess,"
    "externalIds,url,publicationDate,fieldsOfStudy"
)


class SemanticScholarClient:
    """Async client for the Semantic Scholar Graph API v1.

    Features:
        - Paper search, retrieval, and citation lookup
        - Transparent rate-limiting (async sleep between requests)
        - Optional API-key authentication via ``x-api-key`` header
        - Graceful error handling (404, 429, network)

    Attributes:
        _config: API configuration instance.
        _base_url: Semantic Scholar API base URL.
        _rate_limit: Requests per second.
        _owns_client: Whether this instance owns the httpx client.
        _client: Underlying httpx async client.
        _headers: Default HTTP headers for requests.
    """

    def __init__(
        self,
        config: APIConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the SemanticScholarClient.

        Args:
            config: API configuration. Defaults to a new ``APIConfig``.
            client: Optional pre-configured httpx client. If None, one
                will be created internally.
        """
        self._config = config or APIConfig()
        self._base_url = self._config.SEMANTIC_SCHOLAR_BASE_URL.rstrip("/")
        self._rate_limit = self._config.SEMANTIC_SCHOLAR_RATE_LIMIT
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
        )
        self._last_request_time: float = 0.0

        # Build default headers
        self._headers: dict[str, str] = {"Accept": "application/json"}
        if self._config.SEMANTIC_SCHOLAR_API_KEY:
            self._headers["x-api-key"] = self._config.SEMANTIC_SCHOLAR_API_KEY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        fields: str | None = None,
    ) -> list[Paper]:
        """Search papers by keyword.

        Args:
            query: Free-text search string.
            limit: Maximum number of results (capped at 100 by the API).
            fields: Comma-separated S2 field names. Defaults to
                :data:`DEFAULT_SEARCH_FIELDS`.

        Returns:
            A list of matching Paper objects.
        """
        fields = fields or DEFAULT_SEARCH_FIELDS
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": fields,
        }

        data = await self._request("GET", "/paper/search", params=params)
        if data is None:
            return []

        raw_papers = data.get("data") or []
        return [
            p for p in (self._parse_paper(rp) for rp in raw_papers) if p is not None
        ]

    async def get_paper(self, paper_id: str) -> Paper | None:
        """Retrieve a single paper by its Semantic Scholar ID or DOI.

        Args:
            paper_id: A Semantic Scholar ``paperId`` or a DOI (e.g.
                ``DOI:10.1234/...``).

        Returns:
            A Paper object, or None if not found.
        """
        data = await self._request(
            "GET",
            f"/paper/{paper_id}",
            params={"fields": DEFAULT_SEARCH_FIELDS},
        )
        if data is None:
            return None
        return self._parse_paper(data)

    async def get_citations(
        self,
        paper_id: str,
        limit: int = 20,
    ) -> list[Paper]:
        """Return papers that cite the given paper.

        Args:
            paper_id: Semantic Scholar paper ID.
            limit: Maximum number of citing papers to return.

        Returns:
            A list of citing Paper objects.
        """
        params: dict[str, Any] = {
            "fields": DEFAULT_SEARCH_FIELDS,
            "limit": min(limit, 100),
        }
        data = await self._request(
            "GET", f"/paper/{paper_id}/citations", params=params
        )
        if data is None:
            return []

        raw_items = data.get("data") or []
        papers: list[Paper] = []
        for item in raw_items:
            # Each item is {"citingPaper": {...}}
            cp = item.get("citingPaper") or {}
            p = self._parse_paper(cp)
            if p is not None:
                papers.append(p)
        return papers

    async def enrich_paper(self, paper: Paper) -> Paper:
        """Enrich an existing Paper with S2 citation count and relevance.

        Uses the paper's ``paper_id``, ``doi``, or ``arxiv_id`` (in that
        priority order) to look up the S2 record.

        Args:
            paper: The Paper object to enrich.

        Returns:
            The same Paper (or a copy) with enriched fields.
        """
        s2_id = self._resolve_s2_paper_id(paper)
        s2_paper = await self.get_paper(s2_id)
        if s2_paper is None:
            return paper

        update = self._build_enrichment_update(paper, s2_paper)
        return paper.model_copy(update=update)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client (only if we own it).

        If the client was provided externally at construction time, this
        is a no-op to avoid closing a shared client.
        """
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> SemanticScholarClient:
        """Enter the async context manager.

        Returns:
            The SemanticScholarClient instance.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the async context manager and close the HTTP client if owned.

        Args:
            exc_type: Exception type, or None if no exception occurred.
            exc_val: Exception instance, or None.
            exc_tb: Traceback object, or None.
        """
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_s2_paper_id(paper: Paper) -> str:
        """Determine the best Semantic Scholar ID for lookup.

        Prefers DOI or ArXiv ID for papers not originating from S2.

        Args:
            paper: The paper to resolve an ID for.

        Returns:
            A string ID suitable for the S2 API.
        """
        if paper.source != PaperSource.SEMANTIC_SCHOLAR and paper.doi:
            return f"DOI:{paper.doi}"
        if paper.source != PaperSource.SEMANTIC_SCHOLAR and paper.arxiv_id:
            return f"ArXiv:{paper.arxiv_id}"
        return paper.paper_id

    @staticmethod
    def _build_enrichment_update(paper: Paper, s2_paper: Paper) -> dict[str, Any]:
        """Build an update dict of fields to enrich from S2 data.

        Args:
            paper: The original paper.
            s2_paper: The paper looked up from Semantic Scholar.

        Returns:
            Dictionary of fields to update on the original paper.
        """
        update: dict[str, Any] = {}
        # Fields from S2 that are always copied when present
        for field in ("citation_count", "relevance_score", "influence_score"):
            val = getattr(s2_paper, field)
            if val is not None:
                update[field] = val
        # Fields only copied when the original paper lacks them
        for field in (
            "abstract", "pdf_url", "url", "doi", "arxiv_id",
            "year", "authors", "keywords",
        ):
            if not getattr(paper, field) and getattr(s2_paper, field):
                update[field] = getattr(s2_paper, field)
        return update

    async def _rate_limit_sleep(self) -> None:
        """Sleep the minimum interval required by the rate limit.

        Calculates the required wait time from ``_rate_limit`` and sleeps
        if the previous request was too recent. Updates
        ``_last_request_time`` after sleeping.
        """
        if self._rate_limit <= 0:
            return
        import time

        now = time.monotonic()
        elapsed = now - self._last_request_time
        wait = (1.0 / self._rate_limit) - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_time = time.monotonic()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Execute an HTTP request with rate-limiting and error handling.

        Args:
            method: HTTP method (e.g. ``"GET"``).
            path: API path (e.g. ``"/paper/search"``).
            params: Optional query parameters.

        Returns:
            Parsed JSON response dict, or None on error.
        """
        await self._rate_limit_sleep()

        url = f"{self._base_url}{path}"
        try:
            response = await self._client.request(
                method,
                url,
                params=params,
                headers=self._headers,
            )
        except httpx.HTTPError as exc:
            logger.warning("Network error contacting S2: %s", exc)
            return None

        if response.status_code == 404:
            logger.debug("S2 resource not found: %s", url)
            return None

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else self._rate_limit * 2
            logger.warning("S2 rate-limited (429). Retrying after %.1fs", wait)
            await asyncio.sleep(wait)
            # Retry once
            try:
                response = await self._client.request(
                    method, url, params=params, headers=self._headers
                )
            except httpx.HTTPError as exc:
                logger.warning("Network error on retry: %s", exc)
                return None
            if response.status_code >= 400:
                logger.error(
                    "S2 returned %d after retry for %s", response.status_code, url
                )
                return None

        elif response.status_code >= 400:
            logger.error(
                "S2 returned %d for %s — %s",
                response.status_code,
                url,
                response.text[:300],
            )
            return None

        try:
            return response.json()
        except Exception:
            logger.error("Failed to parse S2 JSON response from %s", url)
            return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_s2_authors(raw: dict[str, Any]) -> list[Author]:
        """Extract author list from raw S2 data.

        Args:
            raw: A dict from the Semantic Scholar API.

        Returns:
            List of Author objects with names and S2 IDs.
        """
        authors: list[Author] = []
        for a in raw.get("authors") or []:
            name = a.get("name")
            if name:
                authors.append(
                    Author(
                        name=name,
                        semantic_scholar_id=a.get("authorId"),
                    )
                )
        return authors

    @staticmethod
    def _parse_s2_pub_date(raw: dict[str, Any]) -> Any:
        """Parse publication date from raw S2 data.

        Args:
            raw: A dict from the Semantic Scholar API.

        Returns:
            A datetime object, or None.
        """
        pub_str = raw.get("publicationDate")
        if not pub_str:
            return None
        try:
            from datetime import date as date_type
            from datetime import datetime as dt

            pub_date = date_type.fromisoformat(pub_str)
            return dt.combine(pub_date, dt.min.time())
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_s2_pdf_url(raw: dict[str, Any]) -> str | None:
        """Extract open-access PDF URL from raw S2 data.

        Args:
            raw: A dict from the Semantic Scholar API.

        Returns:
            PDF URL string, or None.
        """
        oa_pdf = raw.get("openAccessPdf")
        if isinstance(oa_pdf, dict) and oa_pdf.get("url"):
            return oa_pdf["url"]
        return None

    @staticmethod
    def _parse_s2_keywords(raw: dict[str, Any]) -> list[str]:
        """Extract keywords from fieldsOfStudy in raw S2 data.

        Args:
            raw: A dict from the Semantic Scholar API.

        Returns:
            List of keyword strings.
        """
        fos = raw.get("fieldsOfStudy")
        if isinstance(fos, list):
            return [f for f in fos if isinstance(f, str)]
        return []

    @staticmethod
    def _parse_paper(raw: dict[str, Any]) -> Paper | None:
        """Convert a raw S2 JSON object into a Paper.

        Args:
            raw: A dict from the Semantic Scholar API.

        Returns:
            A Paper object, or None if the data is insufficient.
        """
        if not raw or not raw.get("paperId"):
            return None

        paper_id = str(raw["paperId"])
        ext = raw.get("externalIds") or {}

        return Paper(
            paper_id=paper_id,
            source=PaperSource.SEMANTIC_SCHOLAR,
            doi=ext.get("DOI"),
            title=raw.get("title") or "",
            abstract=raw.get("abstract"),
            authors=SemanticScholarClient._parse_s2_authors(raw),
            published_date=SemanticScholarClient._parse_s2_pub_date(raw),
            year=raw.get("year"),
            citation_count=raw.get("citationCount"),
            url=raw.get("url"),
            pdf_url=SemanticScholarClient._parse_s2_pdf_url(raw),
            arxiv_id=ext.get("ArXiv"),
            keywords=SemanticScholarClient._parse_s2_keywords(raw),
        )
