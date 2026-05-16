"""Semantic Scholar Graph API v1 client."""

from __future__ import annotations

import asyncio
import logging
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
    """

    def __init__(
        self,
        config: APIConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
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

        Parameters
        ----------
        query:
            Free-text search string.
        limit:
            Maximum number of results (capped at 100 by the API).
        fields:
            Comma-separated S2 field names.  Defaults to
            :data:`DEFAULT_SEARCH_FIELDS`.
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

        Parameters
        ----------
        paper_id:
            A Semantic Scholar ``paperId`` or a DOI (e.g.
            ``DOI:10.1234/...``).
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

        Parameters
        ----------
        paper_id:
            Semantic Scholar paper ID.
        limit:
            Maximum number of citing papers to return.
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
        """
        s2_id = paper.paper_id
        # Prefer DOI-based lookup when the paper didn't originate from S2
        if paper.source != PaperSource.SEMANTIC_SCHOLAR and paper.doi:
            s2_id = f"DOI:{paper.doi}"
        elif paper.source != PaperSource.SEMANTIC_SCHOLAR and paper.arxiv_id:
            s2_id = f"ArXiv:{paper.arxiv_id}"

        s2_paper = await self.get_paper(s2_id)
        if s2_paper is None:
            return paper

        # Merge fields that S2 can fill
        update: dict[str, Any] = {}
        if s2_paper.citation_count is not None:
            update["citation_count"] = s2_paper.citation_count
        if s2_paper.relevance_score is not None:
            update["relevance_score"] = s2_paper.relevance_score
        if s2_paper.influence_score is not None:
            update["influence_score"] = s2_paper.influence_score
        if paper.abstract is None and s2_paper.abstract:
            update["abstract"] = s2_paper.abstract
        if paper.pdf_url is None and s2_paper.pdf_url:
            update["pdf_url"] = s2_paper.pdf_url
        if paper.url is None and s2_paper.url:
            update["url"] = s2_paper.url
        if paper.year is None and s2_paper.year is not None:
            update["year"] = s2_paper.year
        if paper.doi is None and s2_paper.doi:
            update["doi"] = s2_paper.doi
        if paper.arxiv_id is None and s2_paper.arxiv_id:
            update["arxiv_id"] = s2_paper.arxiv_id
        if not paper.authors and s2_paper.authors:
            update["authors"] = s2_paper.authors
        if not paper.keywords and s2_paper.keywords:
            update["keywords"] = s2_paper.keywords

        return paper.model_copy(update=update)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client (only if we own it)."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> SemanticScholarClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _rate_limit_sleep(self) -> None:
        """Sleep the minimum interval required by the rate limit."""
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
        """Execute an HTTP request with rate-limiting and error handling."""
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
    def _parse_paper(raw: dict[str, Any]) -> Paper | None:
        """Convert a raw S2 JSON object into a :class:`Paper`."""
        if not raw or not raw.get("paperId"):
            return None

        paper_id = str(raw["paperId"])
        title = raw.get("title") or ""

        # Authors
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

        # External IDs
        ext = raw.get("externalIds") or {}
        doi = ext.get("DOI")
        arxiv_id = ext.get("ArXiv")

        # PDF URL
        pdf_url: str | None = None
        oa_pdf = raw.get("openAccessPdf")
        if isinstance(oa_pdf, dict) and oa_pdf.get("url"):
            pdf_url = oa_pdf["url"]

        # Publication date -> datetime
        pub_date: Any = None
        pub_str = raw.get("publicationDate")
        if pub_str:
            try:
                from datetime import date as date_type

                pub_date = date_type.fromisoformat(pub_str)
                # Convert to datetime at midnight
                from datetime import datetime as dt

                pub_date = dt.combine(pub_date, dt.min.time())
            except (ValueError, TypeError):
                pub_date = None

        # Fields of study -> keywords
        keywords: list[str] = []
        fos = raw.get("fieldsOfStudy")
        if isinstance(fos, list):
            keywords = [f for f in fos if isinstance(f, str)]

        return Paper(
            paper_id=paper_id,
            source=PaperSource.SEMANTIC_SCHOLAR,
            doi=doi,
            title=title,
            abstract=raw.get("abstract"),
            authors=authors,
            published_date=pub_date,
            year=raw.get("year"),
            citation_count=raw.get("citationCount"),
            url=raw.get("url"),
            pdf_url=pdf_url,
            arxiv_id=arxiv_id,
            keywords=keywords,
        )
