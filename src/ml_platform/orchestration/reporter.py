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
    """Aggregated report for a single paper across all pipeline stages.

    Attributes:
        paper: The paper entity.
        processing_success: Whether text processing succeeded.
        processing_duration: Processing stage duration in seconds.
        processing_error: Error message if processing failed.
        codegen_success: Whether code generation succeeded.
        codegen_files: List of generated code file paths.
        codegen_output_dir: Directory containing generated code.
        codegen_repo_url: URL of the GitHub repository with generated code.
        codegen_duration: Code generation duration in seconds.
        codegen_error: Error message if code generation failed.
        notion_page_url: URL of the created Notion page.
        notion_page_id: ID of the created Notion page.
    """

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
        """Total time across processing and codegen stages."""
        return self.processing_duration + self.codegen_duration


@dataclass
class PipelineRunSummary:
    """Summary of an entire pipeline run (discovery → processing → codegen).

    Attributes:
        run_id: Unique identifier for this run.
        started_at: Timestamp when the run started.
        finished_at: Timestamp when the run finished.
        discovery_queries: Search queries used for discovery.
        total_papers_discovered: Total number of papers found.
        discovery_duration: Discovery stage duration in seconds.
        papers_processed: Number of papers successfully processed.
        papers_failed: Number of papers that failed processing.
        processing_duration: Processing stage duration in seconds.
        codegen_attempts: Number of code generation attempts.
        codegen_successes: Number of successful code generations.
        codegen_duration: Code generation stage duration in seconds.
        reports: Per-paper report details.
    """

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
        """Total wall-clock time across all stages."""
        return self.discovery_duration + self.processing_duration + self.codegen_duration

    @property
    def success_rate(self) -> float:
        """Fraction of papers that were successfully processed."""
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

    Attributes:
        api_key: Notion API key used for authentication.
        parent_page_id: ID of the parent Notion page.
    """

    def __init__(
        self,
        api_key: str | None = None,
        parent_page_id: str | None = None,
    ) -> None:
        """Initialize the Notion reporter.

        Args:
            api_key: Notion API key. Defaults to the ``NOTION_API_KEY``
                environment variable.
            parent_page_id: Notion parent page ID. Defaults to the
                ``NOTION_PARENT_PAGE_ID`` environment variable.
        """
        self.api_key = api_key or NOTION_API_KEY
        self.parent_page_id = parent_page_id or NOTION_PARENT_PAGE_ID
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> NotionReporter:
        """Enter the async context and create the HTTP client.

        Returns:
            The ``NotionReporter`` instance with an active HTTP client.
        """
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

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the async context and close the HTTP client.

        Args:
            exc_type: Exception type, or None.
            exc_val: Exception value, or None.
            exc_tb: Traceback object, or None.
        """
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the active HTTP client.

        Returns:
            The initialised ``httpx.AsyncClient``.

        Raises:
            RuntimeError: If the reporter is not used as an async context
                manager.
        """
        if self._client is None:
            raise RuntimeError("NotionReporter must be used as async context manager")
        return self._client

    # ── Public API ─────────────────────────────────────────────────────────

    async def create_paper_page(self, report: PaperReport) -> dict:
        """Create a Notion page for a single paper report.

        Args:
            report: Aggregated paper report with processing & codegen results.

        Returns:
            Notion API response dict (contains ``'id'`` and ``'url'``).

        Raises:
            httpx.HTTPStatusError: If the Notion API returns an error.
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

        Raises:
            httpx.HTTPStatusError: If the Notion API returns an error.
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
        """Build Notion block children for a paper page.

        Args:
            report: The paper report to render.

        Returns:
            A list of Notion block dicts.
        """
        blocks: list[dict] = []
        blocks.extend(self._build_metadata_section(report))
        blocks.extend(self._build_metrics_section(report))
        blocks.extend(self._build_processing_section(report))
        blocks.extend(self._build_codegen_section(report))
        blocks.extend(self._build_footer_section(report))
        return blocks

    def _build_metadata_section(self, report: PaperReport) -> list[dict]:
        """Build Notion blocks for the paper metadata section.

        Args:
            report: The paper report containing paper metadata.

        Returns:
            A list of Notion block dicts for the metadata section.
        """
        paper = report.paper
        blocks: list[dict] = []
        blocks.append(self._heading2("Paper Metadata"))

        author_names = ", ".join(a.name for a in paper.authors) or "Unknown"
        blocks.append(self._paragraph(f"**Authors:** {author_names}"))

        if paper.arxiv_id:
            blocks.append(
                self._paragraph(
                    f"**arXiv:** [https://arxiv.org/abs/{paper.arxiv_id}]"
                    f"(https://arxiv.org/abs/{paper.arxiv_id})"
                )
            )

        if paper.doi:
            blocks.append(self._paragraph(f"**DOI:** {paper.doi}"))

        if paper.published_date:
            blocks.append(
                self._paragraph(
                    f"**Published:** {paper.published_date.strftime('%Y-%m-%d')}"
                )
            )

        blocks.append(
            self._paragraph(
                f"**Source:** {paper.source.value}  |  "
                f"**Categories:** {', '.join(paper.categories) or 'N/A'}"
            )
        )
        return blocks

    def _build_metrics_section(self, report: PaperReport) -> list[dict]:
        """Build Notion blocks for the metrics section.

        Args:
            report: The paper report containing paper metrics.

        Returns:
            A list of Notion block dicts for the metrics section.
        """
        paper = report.paper
        blocks: list[dict] = []
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
        return blocks

    def _build_processing_section(self, report: PaperReport) -> list[dict]:
        """Build Notion blocks for the processing result section.

        Args:
            report: The paper report containing processing results.

        Returns:
            A list of Notion block dicts for the processing section.
        """
        paper = report.paper
        blocks: list[dict] = []
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
        return blocks

    def _build_codegen_section(self, report: PaperReport) -> list[dict]:
        """Build Notion blocks for the code generation result section.

        Args:
            report: The paper report containing codegen results.

        Returns:
            A list of Notion block dicts for the codegen section.
        """
        blocks: list[dict] = []
        blocks.append(self._heading2("Code Generation"))

        if report.codegen_success:
            blocks.extend(self._build_codegen_success_blocks(report))
        elif report.codegen_error:
            blocks.append(self._paragraph("**Status:** FAILED"))
            blocks.append(self._paragraph(f"**Error:** {report.codegen_error}"))
        else:
            blocks.append(self._paragraph("**Status:** Not attempted"))

        if report.codegen_duration > 0:
            blocks.append(
                self._paragraph(f"**Codegen duration:** {report.codegen_duration:.1f}s")
            )
        return blocks

    def _build_codegen_success_blocks(self, report: PaperReport) -> list[dict]:
        """Build Notion blocks for a successful code generation result.

        Args:
            report: The paper report with successful codegen.

        Returns:
            A list of Notion block dicts for the success details.
        """
        blocks: list[dict] = []
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
        if report.codegen_files:
            file_lines = "\n".join(f"  - `{f}`" for f in report.codegen_files[:30])
            blocks.append(self._paragraph(f"**Generated files:**\n{file_lines}"))
            if len(report.codegen_files) > 30:
                blocks.append(
                    self._paragraph(
                        f"  ... and {len(report.codegen_files) - 30} more files"
                    )
                )
        return blocks

    def _build_footer_section(self, report: PaperReport) -> list[dict]:
        """Build Notion blocks for the page footer with timing.

        Args:
            report: The paper report with duration info.

        Returns:
            A list of Notion block dicts for the footer.
        """
        blocks: list[dict] = []
        blocks.append(self._divider())
        blocks.append(
            self._paragraph(
                f"**Total pipeline time:** {report.total_duration:.1f}s  |  "
                f"Reported at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
        )
        return blocks

    def _build_summary_children(self, summary: PipelineRunSummary) -> list[dict]:
        """Build Notion block children for a pipeline run summary page.

        Args:
            summary: The pipeline run summary to render.

        Returns:
            A list of Notion block dicts.
        """
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
        """Create a Notion heading_1 block.

        Args:
            text: Heading text content.

        Returns:
            A Notion block dict.
        """
        return {
            "object": "block",
            "type": "heading_1",
            "heading_1": {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    @staticmethod
    def _heading2(text: str) -> dict:
        """Create a Notion heading_2 block.

        Args:
            text: Heading text content.

        Returns:
            A Notion block dict.
        """
        return {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    @staticmethod
    def _paragraph(text: str) -> dict:
        """Create a paragraph block.

        Notion rich_text content limit is 2000 characters. Text exceeding
        the limit is truncated to the first 2000 characters.

        Args:
            text: Paragraph text content.

        Returns:
            A Notion paragraph block dict.
        """
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
        """Create a Notion divider block.

        Returns:
            A Notion divider block dict.
        """
        return {"object": "block", "type": "divider", "divider": {}}


# ── Discord / notification helpers ────────────────────────────────────────────

def _discord_author_line(authors: list) -> str | None:
    """Format an author list for Discord display.

    Args:
        authors: List of Author objects.

    Returns:
        Formatted author string, or None if no authors.
    """
    if not authors:
        return None
    author_str = ", ".join(a.name for a in authors[:5])
    if len(authors) > 5:
        author_str += f" +{len(authors) - 5} more"
    return f"**Authors:** {author_str}"


def _discord_codegen_line(report: PaperReport) -> str | None:
    """Format the codegen status line for Discord.

    Args:
        report: Aggregated paper report.

    Returns:
        Formatted codegen line, or None if not applicable.
    """
    if report.codegen_success:
        file_count = len(report.codegen_files)
        repo_str = f" | **Repo:** {report.codegen_repo_url}" if report.codegen_repo_url else ""
        return f"**Codegen:** {file_count} files generated ({report.codegen_duration:.0f}s){repo_str}"
    if report.codegen_error:
        return f"**Codegen:** Failed — {report.codegen_error[:100]}"
    return None


def format_discord_paper_message(report: PaperReport) -> str:
    """Format a concise Discord notification for a single paper result.

    Args:
        report: Aggregated paper report.

    Returns:
        A string suitable for Hermes ``send_message``.
    """
    paper = report.paper
    lines: list[str] = []

    icon = "SUCCESS" if report.processing_success else "FAILED"
    lines.append(f"**Paper Processed: {icon}**")

    title = paper.title[:150] + ("..." if len(paper.title) > 150 else "")
    lines.append(f"**Title:** {title}")

    author_line = _discord_author_line(paper.authors)
    if author_line:
        lines.append(author_line)

    if paper.arxiv_id:
        lines.append(f"**arXiv:** https://arxiv.org/abs/{paper.arxiv_id}")

    citations = str(paper.citation_count) if paper.citation_count is not None else "N/A"
    code = "Yes" if paper.has_code else "No"
    lines.append(f"**Citations:** {citations} | **Existing code:** {code}")

    codegen_line = _discord_codegen_line(report)
    if codegen_line:
        lines.append(codegen_line)

    lines.append(f"**Total time:** {report.total_duration:.1f}s")

    if report.notion_page_url:
        lines.append(f"**Notion:** {report.notion_page_url}")

    return "\n".join(lines)


def format_discord_summary_message(summary: PipelineRunSummary) -> str:
    """Format a concise Discord notification for a full pipeline run summary.

    Args:
        summary: Aggregated pipeline run summary.

    Returns:
        A Discord-formatted summary string.
    """
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

    This is a convenience wrapper around ``NotionReporter.create_paper_page()``.

    Args:
        report: Paper report to publish.
        api_key: Notion API key (defaults to env var).
        parent_page_id: Notion parent page ID.

    Returns:
        The same ``PaperReport`` with ``notion_page_url`` populated.
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

    Also populates ``notion_page_url`` on each ``PaperReport`` so Discord
    notifications can include the Notion link.

    Args:
        summary: Pipeline run summary with reports populated.
        notion_api_key: Notion API key.
        notion_parent_page_id: Notion parent page ID.

    Returns:
        The same ``PipelineRunSummary`` with reports updated.
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
