"""PapersWithCode API client for checking code availability and discovering papers."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from ml_platform.config import APIConfig
from ml_platform.models import Author, Paper, PaperSource

logger = logging.getLogger(__name__)


class PapersWithCodeClient:
    """Async client for the PapersWithCode API v1.

    Primary purpose: check whether a paper already has a public code
    implementation and, if so, surface the most relevant repository.

    API docs: https://paperswithcode.com/api/v1/
    """

    def __init__(
        self,
        base_url: str | None = None,
        rate_limit: float | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = (base_url or APIConfig.PWC_BASE_URL).rstrip("/")
        self._rate_limit = rate_limit if rate_limit is not None else APIConfig.PWC_RATE_LIMIT
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Accept": "application/json"},
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> PapersWithCodeClient:
        await self._ensure_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _throttle(self) -> None:
        """Sleep if needed to stay within the configured rate limit."""
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

        Returns parsed JSON dict, or None on non-recoverable errors.
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
        """Paginate through PWC 'next'-cursor responses until *limit* items collected."""
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
        """Search papers whose title/abstract match *query*.

        Returns a list of Paper objects enriched with the best available
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

        PWC exposes ``/papers/arxiv:{arxiv_id}/``.  If it exists we then
        look up associated repositories.
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
        """Check whether a public code implementation exists for *arxiv_id*.

        Returns::

            {
                "has_code": bool,
                "code_url": str | None,
                "repo_name": str | None,
                "framework": str | None,
            }
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

        Each entry is a dict with at least ``id`` and ``name`` keys.
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

        Returns papers sorted by the PWC ranking for that task.
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
        """Build a Paper model from a PWC JSON object."""
        if not raw:
            return None

        title = raw.get("title") or raw.get("name")
        if not title:
            return None

        paper_id = raw.get("id") or raw.get("arxiv_id") or ""
        arxiv_id = raw.get("arxiv_id")

        # Authors — PWC may return a list of strings or dicts
        authors: list[Author] = []
        for a in raw.get("authors") or []:
            if isinstance(a, str):
                authors.append(Author(name=a))
            elif isinstance(a, dict):
                authors.append(Author(name=a.get("name", "")))

        # Published date
        published_date = raw.get("published") or raw.get("published_date")

        return Paper(
            paper_id=arxiv_id or paper_id,
            source=PaperSource.PAPERSWITHCODE,
            title=title,
            abstract=raw.get("abstract"),
            authors=authors,
            url=raw.get("url_abs") or raw.get("url"),
            pdf_url=raw.get("url_pdf"),
            arxiv_id=arxiv_id,
            pwc_id=paper_id if paper_id != (arxiv_id or "") else None,
            published_date=published_date,
            citation_count=raw.get("citation_count"),
        )

    @staticmethod
    def _pick_best_repo(repos: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Select the most relevant repository from a list of code repos.

        Priority:
          1. Official repos (``is_official`` flag)
          2. Highest star count
          3. First available
        """
        if not repos:
            return None

        official = [r for r in repos if r.get("is_official")]
        pool = official if official else repos

        # Sort by stars descending
        pool_sorted = sorted(pool, key=lambda r: r.get("stars", 0) or 0, reverse=True)

        return pool_sorted[0] if pool_sorted else None
