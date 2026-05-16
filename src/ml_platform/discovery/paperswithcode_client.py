"""ML Research Platform — PapersWithCode API client.

Async client for checking code availability and discovering papers
via the PapersWithCode API v1.
"""

from __future__ import annotations

import asyncio
import logging
import re
import types
from typing import Any

import httpx

from ml_platform.config import APIConfig
from ml_platform.models import Author, Paper, PaperSource

logger = logging.getLogger(__name__)


class PapersWithCodeClient:
    """Async client for the PapersWithCode API v1.

    Primary purpose: check whether a paper already has a public code
    implementation and, if so, surface the most relevant repository.

    Attributes:
        _base_url: API base URL.
        _rate_limit: Requests per second.
        _timeout: HTTP request timeout in seconds.
        _client: Underlying httpx async client.
        _last_request_time: Timestamp of last request for rate limiting.
    """

    def __init__(
        self,
        base_url: str | None = None,
        rate_limit: float | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the PapersWithCodeClient.

        Args:
            base_url: API base URL. Defaults to ``APIConfig.PWC_BASE_URL``.
            rate_limit: Requests per second. Defaults to
                ``APIConfig.PWC_RATE_LIMIT``.
            timeout: HTTP request timeout in seconds.
        """
        self._base_url = (base_url or APIConfig.PWC_BASE_URL).rstrip("/")
        self._rate_limit = rate_limit if rate_limit is not None else APIConfig.PWC_RATE_LIMIT
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Return the active httpx client, creating one if needed.

        Returns:
            The active httpx.AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Accept": "application/json"},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client.

        This method is idempotent; calling it when the client is already
        closed (or was never opened) is a no-op.
        """
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> PapersWithCodeClient:
        """Enter the async context manager and initialise the HTTP client.

        Returns:
            The PapersWithCodeClient instance.
        """
        await self._ensure_client()
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
        await self.close()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _throttle(self) -> None:
        """Sleep if needed to stay within the configured rate limit.

        Calculates the minimum interval between requests from
        ``_rate_limit`` and sleeps for the remaining time if the
        previous request was too recent.
        """
        if self._rate_limit <= 0:
            return
        import time

        now = time.monotonic()
        elapsed = now - self._last_request_time
        min_interval = 1.0 / self._rate_limit
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Low-level request helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Perform a GET request with rate-limiting and error handling.

        Args:
            path: API path to request.
            params: Optional query parameters.

        Returns:
            Parsed JSON dict, or None on non-recoverable errors.
        """
        client = await self._ensure_client()
        await self._throttle()
        try:
            resp = await client.get(path, params=params)
        except httpx.HTTPError as exc:
            logger.warning("PWC request failed: %s %s — %s", "GET", path, exc)
            return None

        if resp.status_code == 404:
            logger.debug("PWC 404 for %s", path)
            return None
        if resp.status_code >= 400:
            logger.warning("PWC HTTP %d for %s", resp.status_code, path)
            return None

        try:
            return resp.json()
        except Exception:
            logger.warning("PWC: failed to parse JSON from %s", path)
            return None

    async def _get_all_pages(self, path: str, params: dict[str, Any] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Paginate through PWC ``next``-cursor responses.

        Args:
            path: API path to request.
            params: Optional query parameters.
            limit: Maximum number of items to collect.

        Returns:
            A list of result dicts.
        """
        if params is None:
            params = {}
        params.setdefault("items_per_page", min(50, limit))

        results: list[dict[str, Any]] = []
        next_token: str | None = None

        while True:
            req_params = {**params}
            if next_token:
                req_params["next"] = next_token

            data = await self._get(path, params=req_params)
            if data is None:
                break

            items = data.get("results") or data.get("papers") or []
            results.extend(items)

            next_token = data.get("next")
            if not next_token or len(results) >= limit:
                break

        return results[:limit]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Search papers whose title/abstract match the query.

        Args:
            query: Free-text search string.
            limit: Maximum number of papers to return.

        Returns:
            A list of Paper objects enriched with the best available
            code repository information.
        """
        logger.info("PWC search: query=%r limit=%d", query, limit)
        raw_papers = await self._get_all_pages(
            "/search/",
            params={"q": query},
            limit=limit,
        )

        papers: list[Paper] = []
        for raw in raw_papers:
            paper = self._parse_paper(raw)
            if paper is not None:
                # Enrich with repo data embedded in search results
                repos = raw.get("repositories") or []
                best = self._pick_best_repo(repos)
                if best:
                    paper.code_url = best.get("url")
                    paper.pwc_id = raw.get("id")
                papers.append(paper)

        return papers

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        """Retrieve a paper by its arXiv ID and determine code availability.

        PWC exposes ``/papers/arxiv:{arxiv_id}/``. If it exists we then
        look up associated repositories.

        Args:
            arxiv_id: An arXiv identifier.

        Returns:
            A Paper object, or None if not found.
        """
        logger.info("PWC get_paper: arxiv_id=%s", arxiv_id)
        # Normalise – strip version suffix for the lookup
        clean_id = re.sub(r"v\d+$", "", arxiv_id)

        data = await self._get(f"/papers/arxiv:{clean_id}/")
        if data is None:
            return None

        paper = self._parse_paper(data)
        if paper is None:
            return None

        # Fetch code repositories for this paper
        code_info = await self.check_code_available(arxiv_id)
        if code_info["has_code"]:
            paper.code_url = code_info["code_url"]

        return paper

    async def check_code_available(self, arxiv_id: str) -> dict[str, Any]:
        """Check whether a public code implementation exists for the arXiv ID.

        Args:
            arxiv_id: An arXiv identifier.

        Returns:
            A dict with keys ``has_code``, ``code_url``, ``repo_name``,
            and ``framework``.
        """
        result: dict[str, Any] = {
            "has_code": False,
            "code_url": None,
            "repo_name": None,
            "framework": None,
        }

        clean_id = re.sub(r"v\d+$", "", arxiv_id)

        # PWC exposes repositories under /repositories/{paper_id}/
        # but we first need the internal paper ID via the arXiv lookup.
        paper_data = await self._get(f"/papers/arxiv:{clean_id}/")
        if paper_data is None:
            return result

        pwc_id = paper_data.get("id")
        if not pwc_id:
            return result

        repos_raw = await self._get_all_pages(
            f"/repositories/{pwc_id}/",
            limit=50,
        )

        if not repos_raw:
            return result

        best = self._pick_best_repo(repos_raw)
        if best:
            result["has_code"] = True
            result["code_url"] = best.get("url")
            result["repo_name"] = best.get("name") or best.get("url", "").split("/")[-1]
            result["framework"] = best.get("framework")

        return result

    async def get_tasks(self) -> list[dict[str, Any]]:
        """Return the list of available ML tasks (e.g. 'image-classification').

        Returns:
            A list of dicts with at least ``id`` and ``name`` keys.
        """
        logger.info("PWC get_tasks")
        raw = await self._get_all_pages("/tasks/", limit=500)
        tasks: list[dict[str, Any]] = []
        for t in raw:
            tasks.append({
                "id": t.get("id"),
                "name": t.get("name"),
                "description": t.get("description"),
            })
        return tasks

    async def get_papers_for_task(self, task: str, limit: int = 10) -> list[Paper]:
        """Get papers evaluated on a specific task (e.g. 'image-classification').

        Args:
            task: An ML task identifier.
            limit: Maximum number of papers to return.

        Returns:
            Papers sorted by the PWC ranking for that task.
        """
        logger.info("PWC get_papers_for_task: task=%r limit=%d", task, limit)
        raw_papers = await self._get_all_pages(
            f"/tasks/{task}/papers/",
            limit=limit,
        )

        papers: list[Paper] = []
        for entry in raw_papers:
            raw = entry.get("paper", entry)  # task endpoints may nest under "paper"
            paper = self._parse_paper(raw)
            if paper is not None:
                # Try to attach repo info if present
                repos = entry.get("repositories") or raw.get("repositories") or []
                best = self._pick_best_repo(repos)
                if best:
                    paper.code_url = best.get("url")
                papers.append(paper)

        return papers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_paper(raw: dict[str, Any]) -> Paper | None:
        """Build a Paper model from a PWC JSON object.

        Args:
            raw: A dict from the PWC API.

        Returns:
            A Paper object, or None if the data is insufficient.
        """
        if not raw:
            return None

        title = raw.get("title") or raw.get("name")
        if not title:
            return None

        paper_id = raw.get("id") or raw.get("arxiv_id") or ""
        arxiv_id = raw.get("arxiv_id")

        return Paper(
            paper_id=arxiv_id or paper_id,
            source=PaperSource.PAPERSWITHCODE,
            title=title,
            abstract=raw.get("abstract"),
            authors=PapersWithCodeClient._parse_pwc_authors(raw),
            url=raw.get("url_abs") or raw.get("url"),
            pdf_url=raw.get("url_pdf"),
            arxiv_id=arxiv_id,
            pwc_id=PapersWithCodeClient._resolve_pwc_id(paper_id, arxiv_id),
            published_date=raw.get("published") or raw.get("published_date"),
            citation_count=raw.get("citation_count"),
        )

    @staticmethod
    def _resolve_pwc_id(paper_id: str, arxiv_id: str | None) -> str | None:
        """Resolve the PapersWithCode ID, returning None if it matches arXiv ID.

        Args:
            paper_id: The paper identifier from PWC.
            arxiv_id: The arXiv identifier, if available.

        Returns:
            The PWC ID, or None if it duplicates the arXiv ID.
        """
        if paper_id == (arxiv_id or ""):
            return None
        return paper_id

    @staticmethod
    def _parse_pwc_authors(raw: dict[str, Any]) -> list[Author]:
        """Parse authors from a PWC JSON object.

        PWC may return authors as a list of strings or dicts.

        Args:
            raw: A dict from the PWC API.

        Returns:
            A list of Author objects.
        """
        authors: list[Author] = []
        for a in raw.get("authors") or []:
            if isinstance(a, str):
                authors.append(Author(name=a))
            elif isinstance(a, dict):
                authors.append(Author(name=a.get("name", "")))
        return authors

    @staticmethod
    def _pick_best_repo(repos: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Select the most relevant repository from a list of code repos.

        Priority:
            1. Official repos (``is_official`` flag)
            2. Highest star count
            3. First available

        Args:
            repos: A list of repository dicts from the PWC API.

        Returns:
            The best repository dict, or None.
        """
        if not repos:
            return None

        official = [r for r in repos if r.get("is_official")]
        pool = official if official else repos

        # Sort by stars descending
        pool_sorted = sorted(pool, key=lambda r: r.get("stars", 0) or 0, reverse=True)

        return pool_sorted[0] if pool_sorted else None
