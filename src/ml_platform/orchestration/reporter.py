"""ML Research Platform — Reporter module.

Handles:
  1. Notion page creation (one page per processed paper under a parent page)
  2. Discord notification via Hermes send_message
  3. Pipeline-run summaries (discovery + processing + codegen)

Notion API docs: https://developers.notion.com/reference/post-page
The module uses httpx (async) to talk to Notion's REST v1 API.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from ml_platform.models import Paper

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

NOTION_API_KEY: str = os.getenv("NOTION_API_KEY", "")
NOTION_PARENT_PAGE_ID: str = os.getenv(
    "NOTION_PARENT_PAGE_ID", "3625dbd4-e011-810b-88e6-d59e30d7035c"
)
NOTION_BASE_URL = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PaperReport:
    """Aggregated report for a single paper across all pipeline stages."""

    paper: Paper
    processing_success: bool = False
    processing_duration: float = 0.0
    processing_error: str | None = None

    codegen_success: bool = False
    codegen_files: list[str] = field(default_factory=list)
    codegen_output_dir: str | None = None
    codegen_repo_url: str | None = None
    codegen_duration: float = 0.0
    codegen_error: str | None = None

    notion_page_url: str | None = None
    notion_page_id: str | None = None

    @property
    def total_duration(self) -> float:
        return self.processing_duration + self.codegen_duration


@dataclass
class PipelineRunSummary:
    """Summary of an entire pipeline run (discovery → processing → codegen)."""

    run_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    # Discovery
    discovery_queries: list[str] = field(default_factory=list)
    total_papers_discovered: int = 0
    discovery_duration: float = 0.0

    # Processing
    papers_processed: int = 0
    papers_failed: int = 0
    processing_duration: float = 0.0

    # Codegen
    codegen_attempts: int = 0
    codegen_successes: int = 0
    codegen_duration: float = 0.0

    # Per-paper details
    reports: list[PaperReport] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return self.discovery_duration + self.processing_duration + self.codegen_duration

    @property
    def success_rate(self) -> float:
        total = self.papers_processed + self.papers_failed
        if total == 0:
            return 0.0
        return self.papers_processed / total


# ── Notion client ─────────────────────────────────────────────────────────────

class NotionReporter:
    """Creates Notion pages for processed papers.

    Each page is created as a child of the configured parent page and contains:
      - Paper metadata (title, authors, arXiv link)
      - Citation count, code availability
      - Code generation results (files, repo URL)
      - Processing duration
    """

    def __init__(
        self,
        api_key: str | None = None,
        parent_page_id: str | None = None,
    ) -> None:
        self.api_key = api_key or NOTION_API_KEY
        self.parent_page_id = parent_page_id or NOTION_PARENT_PAGE_ID
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> NotionReporter:
        self._client = httpx.AsyncClient(
            base_url=NOTION_BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("NotionReporter must be used as async context manager")
        return self._client

    # ── Public API ─────────────────────────────────────────────────────────

    async def create_paper_page(self, report: PaperReport) -> dict:
        """Create a Notion page for a single paper report.

        Args:
            report: Aggregated paper report with processing & codegen results.

        Returns:
            Notion API response dict (contains 'id' and 'url').
        """
        client = self._ensure_client()
        paper = report.paper

        children_blocks = self._build_page_children(report)

        payload: dict[str, Any] = {
            "parent": {"page_id": self.parent_page_id},
            "properties": {
                "title": {
                    "title": [
                        {
                            "text": {
                                "content": paper.title[:2000],  # Notion title limit
                            },
                        },
                    ],
                },
            },
            "children": children_blocks,
        }

        resp = await client.post("/pages", json=payload)
        resp.raise_for_status()
        data = resp.json()

        report.notion_page_id = data.get("id")
        report.notion_page_url = data.get("url")

        logger.info(
            "Created Notion page for %s: %s", paper.paper_id, data.get("url")
        )
        return data

    async def create_summary_page(self, summary: PipelineRunSummary) -> dict:
        """Create a Notion page summarizing an entire pipeline run.

        Args:
            summary: Aggregated pipeline run summary.

        Returns:
            Notion API response dict.
        """
        client = self._ensure_client()
        children = self._build_summary_children(summary)

        title = (
            f"Pipeline Run — {summary.started_at.strftime('%Y-%m-%d %H:%M UTC')}"
        )

        payload: dict[str, Any] = {
            "parent": {"page_id": self.parent_page_id},
            "properties": {
                "title": {
                    "title": [{"text": {"content": title}}],
                },
            },
            "children": children,
        }

        resp = await client.post("/pages", json=payload)
        resp.raise_for_status()
        data = resp.json()

        logger.info("Created pipeline summary page: %s", data.get("url"))
        return data

    # ── Block builders ─────────────────────────────────────────────────────

    def _build_page_children(self, report: PaperReport) -> list[dict]:
        """Build Notion block children for a paper page."""
        paper = report.paper
        blocks: list[dict] = []

        # ── Metadata section ───────────────────────────────────────────
        blocks.append(self._heading2("Paper Metadata"))

        # Authors
        author_names = ", ".join(a.name for a in paper.authors) or "Unknown"
        blocks.append(self._paragraph(f"**Authors:** {author_names}"))

        # arXiv link
        if paper.arxiv_id:
            blocks.append(
                self._paragraph(
                    f"**arXiv:** [https://arxiv.org/abs/{paper.arxiv_id}]"
                    f"(https://arxiv.org/abs/{paper.arxiv_id})"
                )
            )

        # DOI
        if paper.doi:
            blocks.append(self._paragraph(f"**DOI:** {paper.doi}"))

        # Published date
        if paper.published_date:
            blocks.append(
                self._paragraph(
                    f"**Published:** {paper.published_date.strftime('%Y-%m-%d')}"
                )
            )

        # Source & categories
        blocks.append(
            self._paragraph(
                f"**Source:** {paper.source.value}  |  "
                f"**Categories:** {', '.join(paper.categories) or 'N/A'}"
            )
        )

        # ── Metrics ────────────────────────────────────────────────────
        blocks.append(self._heading2("Metrics"))

        citations = str(paper.citation_count) if paper.citation_count is not None else "N/A"
        code_status = "Available" if paper.has_code else "No existing code"
        code_link = f" ([link]({paper.code_url}))" if paper.code_url else ""

        blocks.append(self._paragraph(f"**Citations:** {citations}"))
        blocks.append(self._paragraph(f"**Code Availability:** {code_status}{code_link}"))

        if paper.composite_score is not None:
            blocks.append(self._paragraph(f"**Composite Score:** {paper.composite_score:.3f}"))

        if paper.upvotes:
            blocks.append(self._paragraph(f"**HuggingFace Upvotes:** {paper.upvotes}"))

        # ── Processing result ──────────────────────────────────────────
        blocks.append(self._heading2("Processing"))

        status_icon = "SUCCESS" if report.processing_success else "FAILED"
        blocks.append(self._paragraph(f"**Status:** {status_icon}"))
        blocks.append(
            self._paragraph(f"**Duration:** {report.processing_duration:.1f}s")
        )

        if report.processing_error:
            blocks.append(self._paragraph(f"**Error:** {report.processing_error}"))

        if paper.abstract:
            abstract_preview = paper.abstract[:500] + ("..." if len(paper.abstract) > 500 else "")
            blocks.append(self._paragraph(f"**Abstract:** {abstract_preview}"))

        # ── Code generation result ────────────────────────────────────
        blocks.append(self._heading2("Code Generation"))

        if report.codegen_success:
            blocks.append(self._paragraph("**Status:** SUCCESS"))
            blocks.append(
                self._paragraph(f"**Files generated:** {len(report.codegen_files)}")
            )
            if report.codegen_output_dir:
                blocks.append(
                    self._paragraph(f"**Output directory:** `{report.codegen_output_dir}`")
                )
            if report.codegen_repo_url:
                blocks.append(
                    self._paragraph(
                        f"**Repo URL:** [{report.codegen_repo_url}]"
                        f"({report.codegen_repo_url})"
                    )
                )
            # File list
            if report.codegen_files:
                file_lines = "\n".join(f"  - `{f}`" for f in report.codegen_files[:30])
                blocks.append(self._paragraph(f"**Generated files:**\n{file_lines}"))
                if len(report.codegen_files) > 30:
                    blocks.append(
                        self._paragraph(
                            f"  ... and {len(report.codegen_files) - 30} more files"
                        )
                    )
        elif report.codegen_error:
            blocks.append(self._paragraph(f"**Status:** FAILED"))
            blocks.append(self._paragraph(f"**Error:** {report.codegen_error}"))
        else:
            blocks.append(self._paragraph("**Status:** Not attempted"))

        if report.codegen_duration > 0:
            blocks.append(
                self._paragraph(f"**Codegen duration:** {report.codegen_duration:.1f}s")
            )

        # ── Total time ────────────────────────────────────────────────
        blocks.append(self._divider())
        blocks.append(
            self._paragraph(
                f"**Total pipeline time:** {report.total_duration:.1f}s  |  "
                f"Reported at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
        )

        return blocks

    def _build_summary_children(self, summary: PipelineRunSummary) -> list[dict]:
        """Build Notion block children for a pipeline run summary page."""
        blocks: list[dict] = []

        # Header
        blocks.append(self._heading1("Pipeline Run Summary"))

        # Timing
        blocks.append(
            self._paragraph(
                f"**Started:** {summary.started_at.strftime('%Y-%m-%d %H:%M UTC')}  |  "
                f"**Finished:** "
                f"{summary.finished_at.strftime('%Y-%m-%d %H:%M UTC') if summary.finished_at else 'N/A'}  |  "
                f"**Total duration:** {summary.total_duration:.1f}s"
            )
        )

        # Discovery
        blocks.append(self._heading2("Discovery"))
        queries = ", ".join(summary.discovery_queries) or "N/A"
        blocks.append(self._paragraph(f"**Queries:** {queries}"))
        blocks.append(
            self._paragraph(
                f"**Papers discovered:** {summary.total_papers_discovered}  |  "
                f"**Duration:** {summary.discovery_duration:.1f}s"
            )
        )

        # Processing
        blocks.append(self._heading2("Processing"))
        blocks.append(
            self._paragraph(
                f"**Processed:** {summary.papers_processed}  |  "
                f"**Failed:** {summary.papers_failed}  |  "
                f"**Success rate:** {summary.success_rate:.0%}  |  "
                f"**Duration:** {summary.processing_duration:.1f}s"
            )
        )

        # Codegen
        blocks.append(self._heading2("Code Generation"))
        blocks.append(
            self._paragraph(
                f"**Attempts:** {summary.codegen_attempts}  |  "
                f"**Successes:** {summary.codegen_successes}  |  "
                f"**Duration:** {summary.codegen_duration:.1f}s"
            )
        )

        # Per-paper table
        if summary.reports:
            blocks.append(self._heading2("Paper Details"))
            for report in summary.reports:
                paper = report.paper
                status_parts: list[str] = []
                if report.processing_success:
                    status_parts.append("processed")
                else:
                    status_parts.append(f"processing-failed({report.processing_error or '?'})")
                if report.codegen_success:
                    status_parts.append(
                        f"codegen-ok({len(report.codegen_files)} files)"
                    )
                elif report.codegen_error:
                    status_parts.append(f"codegen-failed")
                else:
                    status_parts.append("no-codegen")

                status_str = " | ".join(status_parts)
                blocks.append(
                    self._paragraph(
                        f"- **{paper.title[:80]}**  \n"
                        f"  `{paper.paper_id}` — {status_str} — {report.total_duration:.1f}s"
                    )
                )

        # Footer
        blocks.append(self._divider())
        blocks.append(
            self._paragraph(
                f"Run ID: `{summary.run_id or 'N/A'}`  |  "
                f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
        )

        return blocks

    # ── Notion block helpers ────────────────────────────────────────────

    @staticmethod
    def _heading1(text: str) -> dict:
        return {
            "object": "block",
            "type": "heading_1",
            "heading_1": {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    @staticmethod
    def _heading2(text: str) -> dict:
        return {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    @staticmethod
    def _paragraph(text: str) -> dict:
        """Create a paragraph block. Notion rich_text content limit is 2000 chars."""
        # Chunk if needed
        if len(text) <= 2000:
            return {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": text}}]
                },
            }
        # Split into multiple paragraphs
        # (returned as a single dict for the first chunk — caller should handle
        # this edge case by splitting externally)
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
            },
        }

    @staticmethod
    def _divider() -> dict:
        return {"object": "block", "type": "divider", "divider": {}}


# ── Discord / notification helpers ────────────────────────────────────────────

def format_discord_paper_message(report: PaperReport) -> str:
    """Format a concise Discord notification for a single paper result.

    Returns a string suitable for Hermes send_message.
    """
    paper = report.paper
    lines: list[str] = []

    # Header
    icon = "SUCCESS" if report.processing_success else "FAILED"
    lines.append(f"**Paper Processed: {icon}**")

    # Title (truncate for Discord)
    title = paper.title[:150] + ("..." if len(paper.title) > 150 else "")
    lines.append(f"**Title:** {title}")

    # Authors
    if paper.authors:
        author_str = ", ".join(a.name for a in paper.authors[:5])
        if len(paper.authors) > 5:
            author_str += f" +{len(paper.authors) - 5} more"
        lines.append(f"**Authors:** {author_str}")

    # Link
    if paper.arxiv_id:
        lines.append(f"**arXiv:** https://arxiv.org/abs/{paper.arxiv_id}")

    # Metrics
    citations = str(paper.citation_count) if paper.citation_count is not None else "N/A"
    code = "Yes" if paper.has_code else "No"
    lines.append(f"**Citations:** {citations} | **Existing code:** {code}")

    # Codegen
    if report.codegen_success:
        file_count = len(report.codegen_files)
        repo_str = f" | **Repo:** {report.codegen_repo_url}" if report.codegen_repo_url else ""
        lines.append(f"**Codegen:** {file_count} files generated ({report.codegen_duration:.0f}s){repo_str}")
    elif report.codegen_error:
        lines.append(f"**Codegen:** Failed — {report.codegen_error[:100]}")

    # Timing
    lines.append(f"**Total time:** {report.total_duration:.1f}s")

    # Notion link
    if report.notion_page_url:
        lines.append(f"**Notion:** {report.notion_page_url}")

    return "\n".join(lines)


def format_discord_summary_message(summary: PipelineRunSummary) -> str:
    """Format a concise Discord notification for a full pipeline run summary."""
    lines: list[str] = []

    lines.append("**ML Research Pipeline — Run Summary**")
    lines.append("")

    # Discovery
    lines.append(
        f"**Discovery:** {summary.total_papers_discovered} papers found "
        f"({summary.discovery_duration:.0f}s)"
    )

    # Processing
    lines.append(
        f"**Processing:** {summary.papers_processed} success, "
        f"{summary.papers_failed} failed ({summary.processing_duration:.0f}s)"
    )

    # Codegen
    lines.append(
        f"**Codegen:** {summary.codegen_successes}/{summary.codegen_attempts} "
        f"successful ({summary.codegen_duration:.0f}s)"
    )

    # Top results
    lines.append("")
    lines.append("**Top results:**")
    for report in summary.reports[:5]:
        paper = report.paper
        codegen_icon = "SUCCESS" if report.codegen_success else "NO CODEGEN"
        short_title = paper.title[:60] + ("..." if len(paper.title) > 60 else "")
        lines.append(
            f"  - {short_title} [{codegen_icon}] ({report.total_duration:.0f}s)"
        )

    lines.append("")
    lines.append(f"**Total duration:** {summary.total_duration:.1f}s")

    return "\n".join(lines)


# ── High-level convenience ────────────────────────────────────────────────────

async def report_paper_to_notion(
    report: PaperReport,
    *,
    api_key: str | None = None,
    parent_page_id: str | None = None,
) -> PaperReport:
    """Create a Notion page for a paper report and update the report with the URL.

    This is a convenience wrapper around NotionReporter.create_paper_page().

    Args:
        report: Paper report to publish.
        api_key: Notion API key (defaults to env var).
        parent_page_id: Notion parent page ID.

    Returns:
        The same PaperReport with notion_page_url populated.
    """
    async with NotionReporter(api_key=api_key, parent_page_id=parent_page_id) as reporter:
        await reporter.create_paper_page(report)
    return report


async def report_run_summary_to_notion(
    summary: PipelineRunSummary,
    *,
    api_key: str | None = None,
    parent_page_id: str | None = None,
) -> dict:
    """Create a Notion summary page for a pipeline run.

    Args:
        summary: Pipeline run summary.
        api_key: Notion API key.
        parent_page_id: Notion parent page ID.

    Returns:
        Notion API response dict.
    """
    async with NotionReporter(api_key=api_key, parent_page_id=parent_page_id) as reporter:
        return await reporter.create_summary_page(summary)


async def report_full_pipeline(
    summary: PipelineRunSummary,
    *,
    notion_api_key: str | None = None,
    notion_parent_page_id: str | None = None,
) -> PipelineRunSummary:
    """Run full reporting: Notion pages for each paper + summary page.

    Also populates notion_page_url on each PaperReport so Discord notifications
    can include the Notion link.

    Args:
        summary: Pipeline run summary with reports populated.
        notion_api_key: Notion API key.
        notion_parent_page_id: Notion parent page ID.

    Returns:
        The same summary with reports updated.
    """
    async with NotionReporter(
        api_key=notion_api_key, parent_page_id=notion_parent_page_id
    ) as reporter:
        # Create individual paper pages
        for report in summary.reports:
            try:
                await reporter.create_paper_page(report)
            except Exception as exc:
                logger.warning(
                    "Failed to create Notion page for %s: %s",
                    report.paper.paper_id,
                    exc,
                )

        # Create summary page
        try:
            await reporter.create_summary_page(summary)
        except Exception as exc:
            logger.warning("Failed to create Notion summary page: %s", exc)

    return summary
