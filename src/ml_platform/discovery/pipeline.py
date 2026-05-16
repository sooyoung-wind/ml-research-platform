"""ML Research Platform — Discovery pipeline orchestrator.

Coordinates paper discovery across multiple sources (arXiv, Semantic
Scholar, HuggingFace), merges and deduplicates results, enriches code
availability, ranks, and stores to the database.
"""

from __future__ import annotations

import asyncio
import time

from ml_platform.config import config
from ml_platform.db import PapersDB
from ml_platform.discovery.arxiv_client import ArxivClient
from ml_platform.discovery.huggingface_client import HuggingFaceClient
from ml_platform.discovery.ranking import merge_and_dedup, rank_papers
from ml_platform.discovery.semantic_scholar_client import SemanticScholarClient
from ml_platform.models import DiscoverResult, Paper


class DiscoveryPipeline:
    """Orchestrates paper discovery across multiple sources.

    Attributes:
        db: PapersDB instance for storage.
        arxiv: ArxivClient instance.
        s2: SemanticScholarClient instance.
        hf: HuggingFaceClient instance.
    """

    def __init__(self) -> None:
        """Initialize the DiscoveryPipeline with default clients."""
        self.db = PapersDB()
        self.arxiv = ArxivClient()
        self.s2 = SemanticScholarClient()
        self.hf = HuggingFaceClient()

    async def search(
        self,
        query: str,
        top_n: int = 10,
        sources: list[str] | None = None,
    ) -> DiscoverResult:
        """Search for papers across sources, rank, and store.

        Args:
            query: Search query.
            top_n: Number of top papers to return.
            sources: Sources to use. None means all. Options:
                ``'arxiv'``, ``'semantic_scholar'``, ``'huggingface'``.

        Returns:
            DiscoverResult with ranked papers.
        """
        start = time.time()
        sources = sources or ["arxiv", "semantic_scholar", "huggingface"]

        # Fetch from all sources in parallel
        tasks = []
        if "arxiv" in sources:
            tasks.append(self._fetch_arxiv(query))
        if "semantic_scholar" in sources:
            tasks.append(self._fetch_s2(query))
        if "huggingface" in sources:
            tasks.append(self._fetch_hf(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        paper_lists: list[list[Paper]] = []
        for r in results:
            if isinstance(r, Exception):
                print(f"  [WARN] Source error: {r}")
            elif isinstance(r, list):
                paper_lists.append(r)

        # Merge and deduplicate
        merged = merge_and_dedup(paper_lists)

        # Enrich with code availability from HuggingFace
        await self._enrich_code_availability(merged)

        # Rank
        ranked = rank_papers(merged, top_n=top_n)

        # Store in DB
        self.db.upsert_papers(ranked)
        self.db.log_discovery(
            query=query,
            total_found=len(merged),
            paper_ids=[p.paper_id for p in ranked],
            duration=time.time() - start,
        )

        return DiscoverResult(
            query=query,
            total_found=len(merged),
            papers=ranked,
            duration_seconds=round(time.time() - start, 2),
        )

    async def daily_discovery(self, top_n: int = 10) -> list[DiscoverResult]:
        """Run daily discovery for all configured topics.

        Args:
            top_n: Number of top papers per topic.

        Returns:
            List of DiscoverResults, one per topic.
        """
        results = []
        for topic in config.DEFAULT_TOPICS:
            print(f"  Discovering: {topic}")
            result = await self.search(query=topic, top_n=top_n)
            results.append(result)
            print(f"    Found {result.total_found} papers, top {len(result.papers)} returned")

        return results

    async def trending(self, limit: int = 20) -> list[Paper]:
        """Fetch today's trending papers from HuggingFace.

        Args:
            limit: Maximum number of papers to return.

        Returns:
            A list of trending Paper objects.
        """
        async with self.hf:
            return await self.hf.get_trending(limit=limit)

    async def _fetch_arxiv(self, query: str) -> list[Paper]:
        """Fetch papers from arXiv.

        Args:
            query: Search query string.

        Returns:
            A list of Paper objects from arXiv.
        """
        async with self.arxiv:
            return await self.arxiv.search_by_keyword(query, max_results=20)

    async def _fetch_s2(self, query: str) -> list[Paper]:
        """Fetch papers from Semantic Scholar.

        Args:
            query: Search query string.

        Returns:
            A list of Paper objects from Semantic Scholar.
        """
        async with self.s2:
            return await self.s2.search(query, limit=20)

    async def _fetch_hf(self, query: str) -> list[Paper]:
        """Fetch papers from HuggingFace.

        Args:
            query: Search query string.

        Returns:
            A list of Paper objects from HuggingFace.
        """
        async with self.hf:
            return await self.hf.search(query, limit=20)

    async def _enrich_code_availability(self, papers: list[Paper]) -> None:
        """Check HuggingFace for existing code implementations.

        Args:
            papers: List of papers to enrich in-place.
        """
        async with self.hf:
            for paper in papers:
                if paper.arxiv_id and not paper.code_url:
                    result = await self.hf.check_code_available(paper.arxiv_id)
                    if result.get("has_code"):
                        paper.code_url = result.get("code_url")
