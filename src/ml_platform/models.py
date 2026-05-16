"""ML Research Platform — Data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PaperSource(str, Enum):
    """Origin of the paper record."""

    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    HUGGINGFACE = "huggingface"
    PAPERSWITHCODE = "paperswithcode"  # deprecated, redirects to HuggingFace
    OPENALEX = "openalex"
    MANUAL = "manual"


class ProcessingStatus(str, Enum):
    """Paper processing pipeline status."""

    DISCOVERED = "discovered"
    PDF_DOWNLOADED = "pdf_downloaded"
    PARSED = "parsed"
    METADATA_ENRICHED = "metadata_enriched"
    CODE_GENERATED = "code_generated"
    VALIDATED = "validated"
    PUSHED = "pushed"
    FAILED = "failed"


class Author(BaseModel):
    """Paper author."""

    name: str
    affiliation: str | None = None
    semantic_scholar_id: str | None = None


class Paper(BaseModel):
    """Unified paper model across all sources."""

    # Identity
    paper_id: str = Field(description="Primary ID (arXiv ID, S2 paper ID, etc.)")
    source: PaperSource
    doi: str | None = None
    title: str
    abstract: str | None = None

    # Authors & metadata
    authors: list[Author] = Field(default_factory=list)
    published_date: datetime | None = None
    venue: str | None = None
    year: int | None = None

    # Metrics
    citation_count: int | None = None
    relevance_score: float | None = None
    influence_score: float | None = None
    upvotes: int = 0  # HuggingFace upvotes

    # Links
    arxiv_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    code_url: str | None = None
    pwc_id: str | None = None  # PapersWithCode ID

    # Content
    categories: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    # Pipeline state
    status: ProcessingStatus = ProcessingStatus.DISCOVERED
    local_pdf_path: str | None = None
    parsed_content: dict | None = None

    # Ranking
    composite_score: float | None = None

    # Timestamps
    discovered_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def has_code(self) -> bool:
        """Whether this paper already has a code implementation."""
        return self.code_url is not None


class DiscoverResult(BaseModel):
    """Result of a paper discovery run."""

    query: str
    total_found: int
    papers: list[Paper]
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_seconds: float = 0.0


class CodeGenResult(BaseModel):
    """Result of a code generation run."""

    paper_id: str
    engine: str  # "papercoder" or "deepcode"
    success: bool
    output_path: str | None = None
    files_generated: list[str] = Field(default_factory=list)
    validation_passed: bool = False
    github_url: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    cost_usd: float | None = None
