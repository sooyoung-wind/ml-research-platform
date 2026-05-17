"""ML Research Platform — Unified Multi-Source Search.

Parallel search across arXiv, Semantic Scholar, HuggingFace, PapersWithCode
with deduplication, ranking, and relevance scoring.

Usage:
    searcher = UnifiedSearcher()
    papers = searcher.search(
        query="retrieval augmented generation hallucination",
        keywords=["RAG", "hallucination", "retrieval"],
        sources=["arxiv", "semantic_scholar", "huggingface", "paperswithcode"],
        max_per_source=10,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ml_platform.models import Paper

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a unified multi-source search.

    Attributes:
        query: The search query used.
        total_found: Total papers found across all sources.
        papers: Deduplicated and ranked papers.
        source_counts: Papers found per source.
        duration: Search duration in seconds.
        errors: Errors encountered per source.
    """
    query: str = ""
    total_found: int = 0
    papers: list[Paper] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    duration: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)


class UnifiedSearcher:
    """Search multiple academic sources in parallel.

    Coordinates searches across:
    - arXiv: Full-text search via HTTP API
    - Semantic Scholar: Graph-based academic search
    - HuggingFace Papers: ML/AI focused paper hub
    - PapersWithCode: Paper-code linking

    Results are deduplicated by arXiv ID / title similarity,
    then ranked by relevance to the query.
    """

    def __init__(self) -> None:
        self._source_map: dict[str, Any] = {}

    def _get_client(self, source: str) -> Any:
        """Lazily load search client."""
        if source not in self._source_map:
            try:
                if source == "arxiv":
                    from ml_platform.discovery.arxiv_client import ArxivClient
                    self._source_map[source] = ArxivClient()
                elif source == "semantic_scholar":
                    from ml_platform.discovery.semantic_scholar_client import SemanticScholarClient
                    self._source_map[source] = SemanticScholarClient()
                elif source == "huggingface":
                    from ml_platform.discovery.huggingface_client import HuggingFaceClient
                    self._source_map[source] = HuggingFaceClient()
                elif source == "paperswithcode":
                    from ml_platform.discovery.paperswithcode_client import PapersWithCodeClient
                    self._source_map[source] = PapersWithCodeClient()
                else:
                    logger.warning(f"Unknown source: {source}")
                    return None
            except Exception as e:
                logger.error(f"Failed to load {source} client: {e}")
                return None
        return self._source_map.get(source)

    async def _search_source(
        self,
        source: str,
        query: str,
        max_results: int,
    ) -> tuple[str, list[Paper], str]:
        """Search a single source. Returns (source, papers, error)."""
        client = self._get_client(source)
        if client is None:
            return source, [], f"Client not available"

        try:
            if source == "arxiv":
                papers = await client.search_by_keyword(query, max_results=max_results)
            elif source == "semantic_scholar":
                papers = await client.search(query, limit=max_results)
            elif source == "huggingface":
                papers = await client.search_papers(query, limit=max_results)
            elif source == "paperswithcode":
                papers = await client.search(query, limit=max_results)
            else:
                papers = []
            return source, papers, ""
        except Exception as e:
            logger.warning(f"Search failed for {source}: {e}")
            return source, [], str(e)

    def search(
        self,
        query: str,
        keywords: list[str] | None = None,
        sources: list[str] | None = None,
        max_per_source: int = 10,
        year_range: tuple[int, int] | None = None,
    ) -> SearchResult:
        """Execute parallel multi-source search.

        Args:
            query: Main search query.
            keywords: Additional search keywords.
            sources: Sources to search (default: all available).
            max_per_source: Max papers from each source.
            year_range: Optional (start_year, end_year) filter.

        Returns:
            SearchResult with deduplicated, ranked papers.
        """
        sources = sources or ["arxiv", "semantic_scholar"]
        # Build enriched query with keywords
        full_query = query
        if keywords:
            full_query = f"{query} {' '.join(keywords[:3])}"

        start_time = time.time()

        # Parallel search across sources
        async def _parallel_search():
            tasks = [
                self._search_source(s, full_query, max_per_source)
                for s in sources
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_parallel_search())

        # Aggregate results
        all_papers: list[Paper] = []
        source_counts: dict[str, int] = {}
        errors: dict[str, str] = {}

        for source, papers, error in results:
            if error:
                errors[source] = error
                source_counts[source] = 0
            else:
                source_counts[source] = len(papers)
                all_papers.extend(papers)

        # Deduplicate
        unique_papers = self._deduplicate(all_papers)

        # Filter by year if specified
        if year_range:
            start_y, end_y = year_range
            unique_papers = [
                p for p in unique_papers
                if p.year and start_y <= p.year <= end_y
            ]

        # Rank by relevance
        ranked_papers = self._rank(unique_papers, full_query)

        duration = time.time() - start_time

        return SearchResult(
            query=full_query,
            total_found=len(ranked_papers),
            papers=ranked_papers,
            source_counts=source_counts,
            duration=duration,
            errors=errors,
        )

    def _deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """Remove duplicate papers based on arXiv ID and title similarity."""
        seen_ids: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[Paper] = []

        for paper in papers:
            # Check by arXiv ID
            arxiv_id = paper.arxiv_id or ""
            if arxiv_id and arxiv_id in seen_ids:
                continue

            # Check by normalized title
            title_norm = (paper.title or "").lower().strip()[:80]
            if title_norm and title_norm in seen_titles:
                continue

            if arxiv_id:
                seen_ids.add(arxiv_id)
            if title_norm:
                seen_titles.add(title_norm)
            unique.append(paper)

        return unique

    def _rank(self, papers: list[Paper], query: str) -> list[Paper]:
        """Simple keyword-based relevance ranking."""
        query_terms = set(query.lower().split())

        def score(paper: Paper) -> int:
            title = (paper.title or "").lower()
            abstract = (paper.abstract or "").lower()
            text = f"{title} {abstract}"
            return sum(1 for t in query_terms if t in text)

        papers.sort(key=score, reverse=True)
        return papers
