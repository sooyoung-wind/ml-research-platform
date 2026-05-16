"""ML Research Platform — Paper ranking system."""

from __future__ import annotations

from datetime import datetime, timedelta

from ml_platform.config import config
from ml_platform.models import Paper


def compute_composite_score(
    paper: Paper,
    max_citations: int = 1000,
    max_freshness_days: int = 365,
) -> float:
    """Compute a composite score for a paper based on configured weights.

    Score = w1 * citation_norm + w2 * relevance_norm + w3 * freshness_norm + w4 * no_code_bonus

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
    import math
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
        top_n: Return only top N papers. None = return all.

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

    When duplicates are found, merge metadata (keep richer data).

    Args:
        paper_lists: Lists of papers from different sources.

    Returns:
        Deduplicated list.
    """
    seen: dict[str, Paper] = {}

    for papers in paper_lists:
        for paper in papers:
            # Primary key: arxiv_id, fallback to paper_id
            key = paper.arxiv_id or f"{paper.source.value}:{paper.paper_id}"

            if key in seen:
                existing = seen[key]
                # Merge: fill in missing fields from the new paper
                if not existing.abstract and paper.abstract:
                    existing.abstract = paper.abstract
                if not existing.citation_count and paper.citation_count:
                    existing.citation_count = paper.citation_count
                if not existing.relevance_score and paper.relevance_score:
                    existing.relevance_score = paper.relevance_score
                if not existing.code_url and paper.code_url:
                    existing.code_url = paper.code_url
                if not existing.pdf_url and paper.pdf_url:
                    existing.pdf_url = paper.pdf_url
                if not existing.doi and paper.doi:
                    existing.doi = paper.doi
                if paper.authors and len(paper.authors) > len(existing.authors):
                    existing.authors = paper.authors
                if paper.keywords and len(paper.keywords) > len(existing.keywords):
                    existing.keywords = paper.keywords
                # Merge categories
                all_cats = list(set(existing.categories + paper.categories))
                existing.categories = all_cats
            else:
                seen[key] = paper

    return list(seen.values())
