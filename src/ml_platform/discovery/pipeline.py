"""ML Research Platform — Discovery pipeline orchestrator."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from ml_platform.config import config
from ml_platform.db import PapersDB
from ml_platform.discovery.arxiv_client import ArxivClient
from ml_platform.discovery.paperswithcode_client import PapersWithCodeClient
from ml_platform.discovery.ranking import merge_and_dedup, rank_papers
from ml_platform.discovery.semantic_scholar_client import SemanticScholarClient
from ml_platform.models import DiscoverResult, Paper, PaperSource


class DiscoveryPipeline:
    """Orchestrates paper discovery across multiple sources."""

    def __init__(self) -> None:
        self.db = PapersDB()
        self.arxiv = ArxivClient()
        self.s2 = SemanticScholarClient()
        self.pwc = PapersWithCodeClient()

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
            sources: Sources to use. None = all. Options: 'arxiv', 'semantic_scholar', 'pwc'.

        Returns:
            DiscoverResult with ranked papers.
        """
        start = time.time()
        sources = sources or ["arxiv", "semantic_scholar", "pwc"]

        # Fetch from all sources in parallel
        tasks = []
        if "arxiv" in sources:
            tasks.append(self._fetch_arxiv(query))
        if "semantic_scholar" in sources:
            tasks.append(self._fetch_s2(query))
        if "pwc" in sources:
            tasks.append(self._fetch_pwc(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        paper_lists: list[list[Paper]] = []
        for r in results:
            if isinstance(r, Exception):
                print(f"  [WARN] Source error: {r}")
            elif isinstance(r, list):
                paper_lists.append(r)

        # Merge and deduplicate
        merged = merge_and_dedup(paper_lists)

        # Enrich with code availability from PWC
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

    async def _fetch_arxiv(self, query: str) -> list[Paper]:
        """Fetch papers from arXiv."""
        async with self.arxiv:
            return await self.arxiv.search_by_keyword(query, max_results=20)

    async def _fetch_s2(self, query: str) -> list[Paper]:
        """Fetch papers from Semantic Scholar."""
        async with self.s2:
            return await self.s2.search(query, limit=20)

    async def _fetch_pwc(self, query: str) -> list[Paper]:
        """Fetch papers from PapersWithCode."""
        async with self.pwc:
            return await self.pwc.search(query, limit=20)

    async def _enrich_code_availability(self, papers: list[Paper]) -> None:
        """Check PapersWithCode for existing code implementations."""
        async with self.pwc:
            for paper in papers:
                if paper.arxiv_id and not paper.code_url:
                    result = await self.pwc.check_code_available(paper.arxiv_id)
                    if result.get("has_code"):
                        paper.code_url = result.get("code_url")
