from __future__ import annotations

import re
from typing import Any

from ml_platform.analysis.models import ReferenceEntry


def extract_references_from_parsed(parsed_content: dict[str, Any] | None) -> list[ReferenceEntry]:
    """Extract references from parsed paper content.
    
    Handles both GROBID-parsed (structured refs) and PyPDF2-parsed (raw text refs).
    
    Args:
        parsed_content: The parsed_content dict from a Paper object.
        
    Returns:
        List of ReferenceEntry objects.
    """
    if not parsed_content:
        return []
    
    # Try GROBID structured references first
    refs = _extract_grobid_refs(parsed_content)
    if refs:
        return refs
    
    # Fall back to raw text extraction
    text = parsed_content.get("full_text", "")
    if not text:
        text = parsed_content.get("raw_text", "")
    if text:
        return _extract_refs_from_text(text)
    
    return []


def _extract_grobid_refs(parsed_content: dict[str, Any]) -> list[ReferenceEntry]:
    """Extract references from GROBID-parsed content."""
    refs = []
    grobid_refs = parsed_content.get("references", [])
    for ref in grobid_refs:
        if isinstance(ref, dict):
            raw = ref.get("raw_text", "")
            title = ref.get("title", "")
            authors = ref.get("authors", "")
            year = _safe_int(ref.get("year"))
            doi = ref.get("doi")
            if doi:
                doi = normalize_doi(doi)
            arxiv_id = detect_arxiv_id(raw)
            refs.append(ReferenceEntry(
                raw_text=raw,
                title=title or None,
                authors=authors or None,
                year=year,
                doi=doi,
                arxiv_id=arxiv_id,
                venue=ref.get("venue") or None,
            ))
    return refs


def _extract_refs_from_text(text: str) -> list[ReferenceEntry]:
    """Extract references from raw text using pattern matching."""
    refs = []
    
    # Look for a references/bibliography section
    ref_section = _find_reference_section(text)
    if not ref_section:
        return refs
    
    # Split into individual references (numbered or bulleted)
    ref_pattern = re.compile(
        r'(?:^\n\[\d+\]|^\n•|^\n\*)\s*(.+?)(?=\n\[\d+\]|\n•|\n\*|$)',
        re.DOTALL | re.MULTILINE,
    )
    matches = ref_pattern.findall(ref_section)
    
    for raw_text in matches:
        raw_text = raw_text.strip()
        if len(raw_text) < 10:  # Skip empty or too-short entries
            continue
        refs.append(ReferenceEntry(
            raw_text=raw_text,
            title=_extract_title(raw_text),
            year=_extract_year(raw_text),
            doi=normalize_doi(raw_text),
            arxiv_id=detect_arxiv_id(raw_text),
        ))
    
    return refs


def _find_reference_section(text: str) -> str | None:
    """Find the references/bibliography section in text."""
    patterns = [
        r'(?i)(?:references|bibliography|works cited)\s*\n(.*)',
        r'(?i)(?:references|bibliography)\s*\n(.*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    return None


def _extract_title(text: str) -> str | None:
    """Try to extract a title from a raw reference string."""
    # Titles are often in quotes or after a period
    match = re.search(r'["""](.+?)["""]', text)
    if match:
        return match.group(1)
    # Try italic-style (common in some formats)
    match = re.search(r'\.\s*([A-Z][^.]+?\.)\s', text)
    if match:
        return match.group(1).strip()
    return None


def _extract_year(text: str) -> int | None:
    """Extract a year from text."""
    match = re.search(r'\b(19|20)\d{2}\b', text)
    if match:
        return int(match.group(0))
    return None


def _safe_int(value: Any) -> int | None:
    """Safely convert value to int."""
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


# ─── Identifier Normalization ───────────────────────────────────────────

_ARXIV_PATTERNS = [
    # New format: YYMM.NNNNN[N]
    re.compile(r'(\d{4}\.\d{4,5})(?:v\d+)?'),
    # Old format: subject-class/NNNNNNNN or subject-class.NNNNNNNN
    re.compile(r'(?:arxiv:)?([a-z-]+\.\d{7})(?:v\d+)?', re.IGNORECASE),
    # With explicit prefix
    re.compile(r'arxiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)', re.IGNORECASE),
]


def detect_arxiv_id(text: str) -> str | None:
    """Detect and extract an arXiv ID from text.
    
    Args:
        text: Text that may contain an arXiv ID.
        
    Returns:
        The extracted arXiv ID (without version), or None.
    """
    for pattern in _ARXIV_PATTERNS:
        match = pattern.search(text)
        if match:
            aid = match.group(1)
            # Strip version suffix
            aid = re.sub(r'v\d+$', '', aid)
            return aid
    return None


def normalize_doi(text: str) -> str | None:
    """Extract and normalize a DOI from text.
    
    Args:
        text: Text that may contain a DOI.
        
    Returns:
        Normalized DOI (lowercase, with 10. prefix), or None.
    """
    match = re.search(r'(?:doi[:\s]*|https?://doi\.org/)?(10\.\d{4,9}/[^\s,;\]]+)', text, re.IGNORECASE)
    if match:
        doi = match.group(1).rstrip('.')
        return doi.lower()
    return None


def canonical_paper_id(
    doi: str | None = None,
    arxiv_id: str | None = None,
    title: str | None = None,
) -> str:
    """Generate a canonical paper identifier for entity resolution.
    
    Priority: DOI > arXiv ID > title-based slug.
    
    Args:
        doi: DOI identifier.
        arxiv_id: arXiv identifier.
        title: Paper title (fallback).
        
    Returns:
        A canonical identifier string.
    """
    if doi:
        return f"doi:{normalize_doi(doi)}"
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if title:
        # Slugify: lowercase, remove punctuation, collapse whitespace
        slug = re.sub(r'[^a-z0-9\s]', '', title.lower())
        slug = re.sub(r'\s+', '_', slug).strip('_')[:80]
        return f"title:{slug}"
    return "unknown"
