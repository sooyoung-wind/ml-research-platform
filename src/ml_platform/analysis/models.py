from __future__ import annotations
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class FiveW1H(BaseModel):
    """Structured 5W1H analysis of a paper."""
    who: str = Field(description="Researchers, affiliations, research groups")
    what: str = Field(description="Research problem, contributions, key claims")
    when: str = Field(description="Publication timeline, relevant temporal context")
    where: str = Field(description="Venue, journal, research institution")
    why: str = Field(description="Motivation, existing limitations, gap being addressed")
    how: str = Field(description="Methodology, datasets, architecture, experimental setup")


class StrengthWeakness(BaseModel):
    """Strengths and weaknesses analysis."""
    strengths: list[str] = Field(default_factory=list, description="Paper strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Paper weaknesses")
    future_work: list[str] = Field(default_factory=list, description="Improvement directions (author-stated + derived)")


class ReferenceEntry(BaseModel):
    """A single reference extracted from a paper."""
    raw_text: str = Field(description="Raw reference text as found in paper")
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None


class EvidenceItem(BaseModel):
    """Evidence sentence backing an analysis claim."""
    claim_type: str = Field(description="Type of claim: who/what/when/where/why/how/strength/weakness/future")
    claim_text: str = Field(description="The analysis claim")
    evidence_text: str = Field(description="Verbatim text from the paper supporting this claim")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score 0-1")


class PaperAnalysis(BaseModel):
    """Complete structured analysis of a research paper."""
    paper_id: str = Field(description="Paper identifier (e.g. arxiv_2312.00752)")
    five_w1h: FiveW1H = Field(description="5W1H structured analysis")
    sw: StrengthWeakness = Field(description="Strengths, weaknesses, future work")
    references: list[ReferenceEntry] = Field(default_factory=list, description="Extracted references")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Evidence sentences")
    summary: str = Field(default="", description="One-paragraph summary")
    key_contributions: list[str] = Field(default_factory=list, description="Key contributions as bullet points")
    methodology_type: str = Field(default="", description="Type: empirical, theoretical, survey, hybrid")
    domain: str = Field(default="", description="Research domain (e.g. computer vision, NLP)")
    status: AnalysisStatus = AnalysisStatus.PENDING
    model_used: str = Field(default="", description="LLM model used for analysis")
    analyzed_at: datetime = Field(default_factory=datetime.now)
    self_correction_applied: bool = Field(default=False)
    correction_notes: str = Field(default="")
