"""HuggingFace Papers API client — replacement for the defunct PapersWithCode API.

PapersWithCode was shut down in August 2025 and redirects to HuggingFace.
The HuggingFace Papers API provides:
  - Paper lookup by arXiv ID (with githubRepo, upvotes, stars)
  - Daily trending papers
  - Paper search via the /api/papers endpoint

API docs (unofficial): https://huggingface.co/docs/hub/api
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from ml_platform.config import APIConfig
from ml_platform.models import Author, Paper, PaperSource

logger = logging.getLogger(__name__)

# Fields we want from the paper detail endpoint
_DEFAULT_FIELDS = "id,title,authors,summary,publishedAt,upvotes,githubRepo,githubStars"


class HuggingFacePapersClient:
    """Async client for the HuggingFace Papers API.

    Features:
    - Paper lookup by arXiv ID (code repo, stars, upvotes)
    - Daily trending papers
    - Rate-limiting via async sleep between requests
    - Graceful error handling (404, 429, network)
    """

    def __init__(
        self,
        config: APIConfig | None = None,
        rate_limit: float | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._config = config or APIConfig()
        self._base_url = self._config.HUGGINGFACE_PAPERS_BASE_URL.rstrip("/")
        self._rate_limit = rate_limit if rate_limit is not None else self._config.HUGGINGFACE_PAPERS_RATE_LIMIT
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    # --- Context manager ---------------------------------------------------

    async def __aenter__(self) -> HuggingFacePapersClient:
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        await self.close()

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # --- Internal helpers ---------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def _throttle(self) -> None:
        """Sleep to respect rate limit."""
        import time
        elapsed = time.monotonic() - self._last_request_time
        min_interval = 1.0 / self._rate_limit if self._rate_limit > 0 else 0
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """GET request with rate-limiting and error handling."""
        client = self._get_client()
        await self._throttle()
        try:
            resp = await client.get(f"{self._base_url}{path}", params=params)
        except httpx.HTTPError as exc:
            logger.warning("HF Papers request failed: %s — %s", path, exc)
            return None

        if resp.status_code == 404:
            logger.debug("HF Papers 404 for %s", path)
            return None
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "2")
            wait = float(retry_after)
            logger.warning("HF Papers rate-limited (429). Retrying after %.1fs", wait)
            await asyncio.sleep(wait)
            try:
                resp = await client.get(f"{self._base_url}{path}", params=params)
            except httpx.HTTPError as exc:
                logger.warning("HF Papers retry failed: %s", exc)
                return None
            if resp.status_code >= 400:
                logger.error("HF Papers returned %d after retry for %s", resp.status_code, path)
                return None
        elif resp.status_code >= 400:
            logger.warning("HF Papers HTTP %d for %s", resp.status_code, path)
            return None

        try:
            return resp.json()
        except Exception:
            logger.warning("HF Papers: failed to parse JSON from %s", path)
            return None

    # --- Public API ---------------------------------------------------------

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        """Look up a paper by arXiv ID. Returns Paper with code availability info.

        The HF Papers API returns githubRepo and githubStars if a repo is linked.
        """
        # Strip version suffix
        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id[-3:] else arxiv_id

        data = await self._get(f"/api/papers/{clean_id}")
        if data is None:
            return None

        return self._parse_paper(data)

    async def check_code_available(self, arxiv_id: str) -> dict[str, Any]:
        """Check if a paper has associated code on HuggingFace.

        Returns dict with: has_code, code_url, repo_name, stars
        """
        result: dict[str, Any] = {
            "has_code": False,
            "code_url": None,
            "repo_name": None,
            "stars": None,
        }

        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id[-3:] else arxiv_id
        data = await self._get(f"/api/papers/{clean_id}")
        if data is None:
            return result

        repo = data.get("githubRepo")
        if repo:
            result["has_code"] = True
            result["code_url"] = repo
            # Extract repo name from URL
            parts = repo.rstrip("/").split("/")
            if len(parts) >= 2:
                result["repo_name"] = f"{parts[-2]}/{parts[-1]}"
            result["stars"] = data.get("githubStars")

        return result

    async def get_trending_papers(self, limit: int = 20) -> list[Paper]:
        """Fetch daily trending papers from HuggingFace.

        Returns up to `limit` papers sorted by upvotes.
        """
        data = await self._get("/api/daily_papers")
        if data is None or not isinstance(data, list):
            return []

        papers: list[Paper] = []
        for item in data[:limit]:
            paper_data = item.get("paper", {})
            paper = self._parse_paper(paper_data)
            if paper:
                # Override published_date with the trending date if available
                published_at = item.get("date") or item.get("publishedAt")
                if published_at:
                    try:
                        paper.published_date = datetime.fromisoformat(
                            published_at.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass
                # Store upvotes in a way we can use for ranking
                paper.upvotes = paper_data.get("upvotes", 0)
                papers.append(paper)

        return papers

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Search papers via HuggingFace.

        Note: HF doesn't have a dedicated search endpoint for papers.
        This falls back to looking up the daily trending papers and filtering
        by keyword match. For full-text search, use arXiv or Semantic Scholar.

        Returns matching papers from trending + recent papers.
        """
        trending = await self.get_trending_papers(limit=100)

        # Simple keyword matching on title and summary
        query_lower = query.lower()
        query_terms = query_lower.split()

        matched: list[Paper] = []
        for paper in trending:
            text = f"{paper.title} {paper.abstract or ''}".lower()
            # Score by number of matching terms
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                paper.relevance_score = score / len(query_terms)
                matched.append(paper)

        matched.sort(key=lambda p: p.relevance_score or 0, reverse=True)
        return matched[:limit]

    # --- Parsing ------------------------------------------------------------

    def _parse_paper(self, data: dict[str, Any]) -> Paper | None:
        """Parse a HuggingFace paper JSON into a Paper model."""
        paper_id = data.get("id")
        if not paper_id:
            return None

        title = data.get("title", "Untitled")

        # Parse authors
        authors: list[Author] = []
        for a in data.get("authors", []):
            name = a.get("name", "")
            if name:
                authors.append(Author(name=name))

        # Parse published date
        published_date = None
        published_at = data.get("publishedAt")
        if published_at:
            try:
                published_date = datetime.fromisoformat(
                    published_at.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        # Code availability
        code_url = data.get("githubRepo")

        # Extract arXiv ID (same as paper id for HF papers)
        arxiv_id = paper_id

        return Paper(
            paper_id=f"hf_{paper_id}",
            arxiv_id=arxiv_id,
            title=title,
            abstract=data.get("summary"),
            authors=authors,
            published_date=published_date,
            source=PaperSource.HUGGINGFACE,
            url=f"https://huggingface.co/papers/{paper_id}",
            pdf_url=f"https://arxiv.org/pdf/{paper_id}",
            code_url=code_url,
            upvotes=data.get("upvotes", 0),
        )
