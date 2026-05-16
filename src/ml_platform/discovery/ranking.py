"""ML Research Platform — Paper ranking system.

Provides composite scoring and ranking of discovered papers based on
citation count, relevance, freshness, and code availability.
"""

from __future__ import annotations

import math
from datetime import datetime

from ml_platform.config import config
from ml_platform.models import Paper


_SIMPLE_MERGE_FIELDS = (
    "abstract",
    "citation_count",
    "relevance_score",
    "code_url",
    "pdf_url",
    "doi",
)


def _merge_paper_fields(existing: Paper, new: Paper) -> None:
    """Merge fields from a new paper into an existing paper in-place.

    Fills in missing simple fields, keeps longer author/keyword lists,
    and merges categories.

    Args:
        existing: The paper to update.
        new: The paper with potentially richer data.
    """
    for field in _SIMPLE_MERGE_FIELDS:
        if not getattr(existing, field) and getattr(new, field):
            setattr(existing, field, getattr(new, field))
    if new.authors and len(new.authors) > len(existing.authors):
        existing.authors = new.authors
    if new.keywords and len(new.keywords) > len(existing.keywords):
        existing.keywords = new.keywords
    all_cats = list(set(existing.categories + new.categories))
    existing.categories = all_cats


def compute_composite_score(
    paper: Paper,
    max_citations: int = 1000,
    max_freshness_days: int = 365,
) -> float:
    """Compute a composite score for a paper based on configured weights.

    Score = w1 * citation_norm + w2 * relevance_norm
            + w3 * freshness_norm + w4 * no_code_bonus

    Args:
        paper: Paper to score.
        max_citations: Assumed max citation count for normalization.
        max_freshness_days: Days window for freshness scoring.

    Returns:
        Composite score between 0 and 1.
    """
    w = config  # shorthand

    # Citation score (0-1, log-scaled)
    cit = paper.citation_count or 0
    citation_score = min(math.log1p(cit) / math.log1p(max_citations), 1.0) if cit > 0 else 0.0

    # Relevance score (already 0-1 from Semantic Scholar)
    relevance_score = paper.relevance_score or 0.0

    # Freshness score (1.0 for today, decaying to 0)
    freshness_score = 0.0
    if paper.published_date:
        pub = paper.published_date
        now = datetime.now(pub.tzinfo) if pub.tzinfo else datetime.now()
        age_days = (now - pub).days
        if age_days >= 0:
            freshness_score = max(0.0, 1.0 - (age_days / max_freshness_days))

    # No-code bonus (papers without existing code get a boost)
    no_code_bonus = 0.0 if paper.has_code else 1.0

    composite = (
        w.WEIGHT_CITATIONS * citation_score
        + w.WEIGHT_RELEVANCE * relevance_score
        + w.WEIGHT_FRESHNESS * freshness_score
        + w.WEIGHT_NO_CODE * no_code_bonus
    )

    return round(composite, 4)


def rank_papers(papers: list[Paper], top_n: int | None = None) -> list[Paper]:
    """Score and rank papers by composite score.

    Args:
        papers: List of papers to rank.
        top_n: Return only top N papers. None means return all.

    Returns:
        Papers sorted by composite_score descending.
    """
    for paper in papers:
        paper.composite_score = compute_composite_score(paper)

    ranked = sorted(papers, key=lambda p: p.composite_score or 0, reverse=True)

    if top_n is not None:
        ranked = ranked[:top_n]

    return ranked


def merge_and_dedup(paper_lists: list[list[Paper]]) -> list[Paper]:
    """Merge papers from multiple sources, deduplicating by arXiv ID.

    When duplicates are found, metadata is merged (keeping richer data).

    Args:
        paper_lists: Lists of papers from different sources.

    Returns:
        Deduplicated list of papers.
    """
    seen: dict[str, Paper] = {}

    for papers in paper_lists:
        for paper in papers:
            # Primary key: arxiv_id, fallback to paper_id
            key = paper.arxiv_id or f"{paper.source.value}:{paper.paper_id}"

            if key in seen:
                _merge_paper_fields(seen[key], paper)
            else:
                seen[key] = paper

    return list(seen.values())
