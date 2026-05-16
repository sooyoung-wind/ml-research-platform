"""ML Research Platform — Metadata enrichment for discovered papers.

Enriches papers with citation counts, code availability, and other metadata
from Semantic Scholar and HuggingFace Papers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ml_platform.discovery.huggingface_client import HuggingFacePapersClient
from ml_platform.discovery.semantic_scholar_client import SemanticScholarClient
from ml_platform.models import Paper, ProcessingStatus

logger = logging.getLogger(__name__)


class MetadataEnricher:
    """Enriches Paper objects with metadata from external APIs.

    Sources:
    - Semantic Scholar: citation count, influential citation count, TLDR, venue
    - HuggingFace Papers: code repo, GitHub stars, upvotes
    """

    def __init__(self) -> None:
        self.s2 = SemanticScholarClient()
        self.hf = HuggingFacePapersClient()

    async def enrich(self, paper: Paper) -> Paper:
        """Enrich a single paper with external metadata.

        Updates citation count, code availability, venue, and other fields.
        """
        tasks = []
        if paper.arxiv_id:
            tasks.append(self._enrich_s2(paper))
            tasks.append(self._enrich_hf(paper))
        elif paper.doi:
            tasks.append(self._enrich_s2_by_doi(paper))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Enrichment error: %s", r)

        if paper.status == ProcessingStatus.DISCOVERED:
            paper.status = ProcessingStatus.METADATA_ENRICHED

        return paper

    async def enrich_batch(self, papers: list[Paper]) -> list[Paper]:
        """Enrich multiple papers sequentially with rate-limiting."""
        enriched = []
        async with self.s2, self.hf:
            for i, paper in enumerate(papers):
                try:
                    enriched.append(await self.enrich(paper))
                except Exception as exc:
                    logger.warning("Failed to enrich %s: %s", paper.paper_id, exc)
                    enriched.append(paper)
                if (i + 1) % 5 == 0:
                    logger.info("Enriched %d/%d papers", i + 1, len(papers))
        return enriched

    async def _enrich_s2(self, paper: Paper) -> None:
        """Fetch citation count, venue, TLDR from Semantic Scholar."""
        if not paper.arxiv_id:
            return

        try:
            paper = await self.s2.enrich_paper(paper)
        except Exception as exc:
            logger.warning("S2 enrichment failed for %s: %s", paper.arxiv_id, exc)

    async def _enrich_s2_by_doi(self, paper: Paper) -> None:
        """Fetch metadata from S2 using DOI."""
        if not paper.doi:
            return

        try:
            paper = await self.s2.enrich_paper(paper)
        except Exception as exc:
            logger.warning("S2 enrichment by DOI failed for %s: %s", paper.doi, exc)

    async def _enrich_hf(self, paper: Paper) -> None:
        """Check code availability via HuggingFace Papers."""
        if not paper.arxiv_id:
            return

        result = await self.hf.check_code_available(paper.arxiv_id)
        if result.get("has_code") and not paper.code_url:
            paper.code_url = result["code_url"]

        # Also get upvotes
        paper_data = await self.hf.get_paper(paper.arxiv_id)
        if paper_data and paper_data.upvotes > paper.upvotes:
            paper.upvotes = paper_data.upvotes
