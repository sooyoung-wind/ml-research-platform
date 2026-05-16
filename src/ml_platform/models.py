"""ML Research Platform — Data models.

Defines Pydantic models for papers, authors, discovery results,
and code generation results used across the platform.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PaperSource(str, Enum):
    """Origin of the paper record.

    Attributes:
        ARXIV: Paper sourced from arXiv.
        SEMANTIC_SCHOLAR: Paper sourced from Semantic Scholar.
        HUGGINGFACE: Paper sourced from HuggingFace Papers.
        PAPERSWITHCODE: Paper sourced from PapersWithCode (deprecated).
        OPENALEX: Paper sourced from OpenAlex.
        MANUAL: Paper added manually.
    """

    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    HUGGINGFACE = "huggingface"
    PAPERSWITHCODE = "paperswithcode"  # deprecated, redirects to HuggingFace
    OPENALEX = "openalex"
    MANUAL = "manual"


class ProcessingStatus(str, Enum):
    """Paper processing pipeline status.

    Attributes:
        DISCOVERED: Paper has been discovered but not processed.
        PDF_DOWNLOADED: PDF has been downloaded.
        PARSED: Paper content has been parsed.
        METADATA_ENRICHED: Metadata has been enriched from external sources.
        CODE_GENERATED: Code implementation has been generated.
        VALIDATED: Generated code has been validated.
        PUSHED: Code has been pushed to GitHub.
        FAILED: Processing failed at some stage.
    """

    DISCOVERED = "discovered"
    PDF_DOWNLOADED = "pdf_downloaded"
    PARSED = "parsed"
    METADATA_ENRICHED = "metadata_enriched"
    CODE_GENERATED = "code_generated"
    VALIDATED = "validated"
    PUSHED = "pushed"
    FAILED = "failed"


class Author(BaseModel):
    """Paper author.

    Attributes:
        name: Author's full name.
        affiliation: Author's institutional affiliation.
        semantic_scholar_id: Semantic Scholar author ID.
    """

    name: str
    affiliation: str | None = None
    semantic_scholar_id: str | None = None


class Paper(BaseModel):
    """Unified paper model across all sources.

    Attributes:
        paper_id: Primary ID (arXiv ID, S2 paper ID, etc.).
        source: Origin of the paper record.
        doi: Digital Object Identifier.
        title: Paper title.
        abstract: Paper abstract text.
        authors: List of paper authors.
        published_date: Date the paper was published.
        venue: Publication venue (conference, journal).
        year: Publication year.
        citation_count: Number of citations.
        relevance_score: Relevance score from search.
        influence_score: Influence score from Semantic Scholar.
        upvotes: HuggingFace upvotes count.
        arxiv_id: arXiv identifier.
        url: URL to the paper page.
        pdf_url: Direct URL to the PDF.
        code_url: URL to the code repository if available.
        pwc_id: PapersWithCode identifier.
        categories: List of arXiv categories.
        keywords: List of paper keywords.
        status: Current processing status.
        local_pdf_path: Local filesystem path to downloaded PDF.
        parsed_content: Parsed content dictionary.
        composite_score: Weighted ranking score.
        discovered_at: Timestamp when the paper was discovered.
        updated_at: Timestamp when the paper was last updated.
    """

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
        """Check whether this paper already has a code implementation.

        Returns:
            True if a code URL is available, False otherwise.
        """
        return self.code_url is not None


class DiscoverResult(BaseModel):
    """Result of a paper discovery run.

    Attributes:
        query: The search query used.
        total_found: Total number of papers found.
        papers: List of discovered Paper objects.
        timestamp: When the discovery was performed.
        duration_seconds: Time taken for the discovery in seconds.
    """

    query: str
    total_found: int
    papers: list[Paper]
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_seconds: float = 0.0


class CodeGenResult(BaseModel):
    """Result of a code generation run.

    Attributes:
        paper_id: Identifier of the paper.
        engine: Code generation engine name.
        success: Whether code generation succeeded.
        output_path: Path to the output directory.
        files_generated: List of generated file paths.
        validation_passed: Whether the generated code passed validation.
        github_url: URL of the pushed GitHub repository.
        error: Error message if generation failed.
        duration_seconds: Time taken for code generation.
        cost_usd: Estimated cost in USD.
    """

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
