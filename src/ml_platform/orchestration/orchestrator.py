"""ML Research Platform — Full pipeline orchestrator.

Ties together all phases:
  Discovery → Processing → Code Generation → GitHub Push → Reporting

Usage:
    orchestrator = ResearchOrchestrator()
    result = await orchestrator.run_paper("2312.00752")
    result = await orchestrator.run_topic("diffusion models", top_n=5)
    result = await orchestrator.run_daily()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ml_platform.models import Paper

logger = logging.getLogger(__name__)


@dataclass
class PaperPipelineResult:
    """Result of running the full pipeline on a single paper.

    Attributes:
        paper_id: arXiv paper identifier.
        paper_title: Title of the paper.
        success: Whether the full pipeline completed successfully.
        error: Error message if the pipeline failed.
        skipped: Whether the paper was skipped.
        downloaded: Whether the PDF was downloaded successfully.
        extracted: Whether text extraction succeeded.
        enriched: Whether metadata enrichment succeeded.
        code_generated: Whether code generation succeeded.
        pushed: Whether the code was pushed to GitHub.
        reported: Whether a Notion report was created.
        pdf_path: Local filesystem path to the downloaded PDF.
        text_chars: Number of characters extracted from the paper.
        code_files: List of generated code file paths.
        code_dir: Directory containing generated code.
        repo_url: URL of the GitHub repository.
        notion_url: URL of the Notion page.
        total_duration: Total pipeline execution time in seconds.
        stage_durations: Mapping of stage name to duration in seconds.
    """

    paper_id: str = ""
    paper_title: str = ""
    success: bool = False
    error: str = ""
    skipped: bool = False

    # Stage results
    downloaded: bool = False
    extracted: bool = False
    enriched: bool = False
    code_generated: bool = False
    pushed: bool = False
    reported: bool = False

    # Outputs
    pdf_path: str = ""
    text_chars: int = 0
    code_files: list[str] = field(default_factory=list)
    code_dir: str = ""
    repo_url: str = ""
    notion_url: str = ""

    # Timing
    total_duration: float = 0.0
    stage_durations: dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineRunResult:
    """Result of running the pipeline on multiple papers.

    Attributes:
        total_papers: Total number of papers attempted.
        successful: Number of papers that completed successfully.
        failed: Number of papers that failed.
        skipped: Number of papers that were skipped.
        results: Per-paper pipeline results.
        total_duration: Total run duration in seconds.
        topic: Topic query that triggered this run.
    """

    total_papers: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[PaperPipelineResult] = field(default_factory=list)
    total_duration: float = 0.0
    topic: str = ""


class ResearchOrchestrator:
    """Full pipeline orchestrator for ML research automation.

    Attributes:
        discover_sources: Discovery sources to search.
        use_grobid: Whether to use GROBID for PDF parsing.
        codegen_mode: Code generation mode (e.g. "optimized").
        codegen_provider: LLM provider for code generation.
        codegen_model: LLM model name for code generation.
        skip_codegen: Whether to skip the code generation stage.
        push_to_github: Whether to push generated code to GitHub.
        github_private: Whether GitHub repos should be private.
        report_to_notion: Whether to create Notion reports.
        report_to_discord: Whether to send Discord notifications.
        max_concurrent: Maximum number of concurrent paper pipelines.
        progress_callback: Optional callback for progress updates.
    """

    def __init__(
        self,
        *,
        discover_sources: list[str] | None = None,
        use_grobid: bool = False,
        codegen_mode: str = "optimized",
        codegen_provider: str = "openai",
        codegen_model: str = "",
        skip_codegen: bool = False,
        push_to_github: bool = True,
        github_private: bool = False,
        report_to_notion: bool = True,
        report_to_discord: bool = False,
        max_concurrent: int = 2,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> None:
        """Initialize the research orchestrator.

        Args:
            discover_sources: Discovery sources to search.
            use_grobid: Whether to use GROBID for PDF parsing.
            codegen_mode: Code generation mode.
            codegen_provider: LLM provider for code generation.
            codegen_model: LLM model name (empty uses provider default).
            skip_codegen: Whether to skip code generation.
            push_to_github: Whether to push to GitHub.
            github_private: Whether repos should be private.
            report_to_notion: Whether to create Notion reports.
            report_to_discord: Whether to send Discord notifications.
            max_concurrent: Maximum concurrent paper pipelines.
            progress_callback: Optional callback ``(stage, pct, msg)``.
        """
        self.discover_sources = discover_sources
        self.use_grobid = use_grobid
        self.codegen_mode = codegen_mode
        self.codegen_provider = codegen_provider
        self.codegen_model = codegen_model
        self.skip_codegen = skip_codegen
        self.push_to_github = push_to_github
        self.github_private = github_private
        self.report_to_notion = report_to_notion
        self.report_to_discord = report_to_discord
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback

    def _progress(self, stage: str, pct: float, msg: str) -> None:
        """Emit a progress update via the callback and logger.

        Args:
            stage: Pipeline stage name.
            pct: Progress percentage (0–100).
            msg: Human-readable progress message.
        """
        if self.progress_callback:
            self.progress_callback(stage, pct, msg)
        logger.info(f"[{stage}] {pct:.0f}% — {msg}")

    async def run_paper(
        self,
        arxiv_id: str,
        *,
        force: bool = False,
    ) -> PaperPipelineResult:
        """Run the full pipeline on a single paper by arXiv ID.

        Executes stages: Download → Extract/Enrich → Code Generation →
        GitHub Push → Notion Report.

        Args:
            arxiv_id: The arXiv paper identifier.
            force: If True, re-download and re-process even if cached.

        Returns:
            A ``PaperPipelineResult`` with per-stage outcomes and outputs.
        """
        start = time.time()
        result = PaperPipelineResult(paper_id=arxiv_id)

        try:
            from ml_platform.models import PaperSource
            paper = Paper(
                paper_id=f"arxiv_{arxiv_id}",
                arxiv_id=arxiv_id,
                title=arxiv_id,
                source=PaperSource.ARXIV,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )

            if not await self._download_stage(arxiv_id, paper, result, force):
                result.total_duration = time.time() - start
                return result

            await self._extract_stage(arxiv_id, paper, result, force)
            await self._codegen_stage(arxiv_id, result)
            await self._push_stage(arxiv_id, result)
            await self._report_stage(arxiv_id, paper, result)

            result.success = True

        except Exception as e:
            result.error = str(e)
            logger.error(f"Pipeline failed for {arxiv_id}: {e}", exc_info=True)

        result.total_duration = time.time() - start
        return result

    async def _download_stage(
        self,
        arxiv_id: str,
        paper: Paper,
        result: PaperPipelineResult,
        force: bool,
    ) -> bool:
        """Execute the PDF download stage.

        Args:
            arxiv_id: The arXiv paper identifier.
            paper: The Paper object to download.
            result: Pipeline result to update.
            force: If True, re-download even if cached.

        Returns:
            True if download succeeded, False otherwise.
        """
        self._progress("download", 10, f"Downloading PDF for {arxiv_id}")
        t0 = time.time()

        from ml_platform.processing.pdf_downloader import PDFDownloader
        async with PDFDownloader() as dl:
            dl_result = await dl.download_paper(paper, force=force)

        if dl_result.success and dl_result.path:
            result.downloaded = True
            result.pdf_path = str(dl_result.path)
            paper.local_pdf_path = result.pdf_path
        else:
            result.error = f"PDF download failed: {dl_result.error}"
            return False

        result.stage_durations["download"] = time.time() - t0
        return True

    async def _extract_stage(
        self,
        arxiv_id: str,
        paper: Paper,
        result: PaperPipelineResult,
        force: bool,
    ) -> None:
        """Execute the text extraction and metadata enrichment stage.

        Args:
            arxiv_id: The arXiv paper identifier.
            paper: The Paper object to process.
            result: Pipeline result to update.
            force: If True, re-process even if cached.
        """
        self._progress("extract", 25, "Extracting text and enriching metadata")
        t0 = time.time()

        from ml_platform.processing.processor import PaperProcessor
        processor = PaperProcessor(use_grobid=self.use_grobid, enrich_metadata=True)
        proc_result = await processor.process_paper(paper, download=False, force=force)

        if proc_result.extracted.get("success"):
            result.extracted = True
            result.text_chars = proc_result.extracted.get("chars", 0)
        if proc_result.enriched:
            result.enriched = True
        result.paper_title = paper.title if paper.title != arxiv_id else ""

        result.stage_durations["extract"] = time.time() - t0

    async def _codegen_stage(
        self,
        arxiv_id: str,
        result: PaperPipelineResult,
    ) -> None:
        """Execute the code generation stage.

        Args:
            arxiv_id: The arXiv paper identifier.
            result: Pipeline result to update.
        """
        if self.skip_codegen:
            self._progress("codegen", 50, "Skipping code generation")
            return

        self._progress("codegen", 50, f"Generating code ({self.codegen_mode} mode)")
        t0 = time.time()

        from ml_platform.codegen.deepcode_runner import DeepCodeConfig, DeepCodeRunner
        dc_config = DeepCodeConfig(
            llm_provider=self.codegen_provider,
            model_name=self.codegen_model or "gpt-4o",
        )
        runner = DeepCodeRunner(dc_config)
        cg_result = await runner.generate(
            result.pdf_path,
            paper_id=arxiv_id,
            paper_title=result.paper_title,
            mode=self.codegen_mode,
        )

        if cg_result.success:
            result.code_generated = True
            result.code_dir = cg_result.output_dir
            result.code_files = cg_result.files_generated
        else:
            logger.warning(f"Code generation failed for {arxiv_id}: {cg_result.error}")

        result.stage_durations["codegen"] = time.time() - t0

    async def _push_stage(
        self,
        arxiv_id: str,
        result: PaperPipelineResult,
    ) -> None:
        """Execute the GitHub push stage.

        Args:
            arxiv_id: The arXiv paper identifier.
            result: Pipeline result to update.
        """
        if not result.code_generated or not self.push_to_github:
            return

        self._progress("push", 75, "Pushing to GitHub")
        t0 = time.time()

        from ml_platform.orchestration.github_pusher import GitHubPusher
        pusher = GitHubPusher()
        visibility = "private" if self.github_private else "public"
        push_result = await pusher.push(
            result.code_dir,
            paper_id=arxiv_id,
            paper_title=result.paper_title,
            visibility=visibility,
        )

        if push_result.success:
            result.pushed = True
            result.repo_url = push_result.repo_url
        else:
            logger.warning(f"GitHub push failed: {push_result.error}")

        result.stage_durations["push"] = time.time() - t0

    async def _report_stage(
        self,
        arxiv_id: str,
        paper: Paper,
        result: PaperPipelineResult,
    ) -> None:
        """Execute the Notion report creation stage.

        Args:
            arxiv_id: The arXiv paper identifier.
            paper: The Paper object.
            result: Pipeline result to update.
        """
        if not self.report_to_notion:
            return

        self._progress("report", 90, "Creating Notion report")
        t0 = time.time()

        try:
            from ml_platform.orchestration.reporter import (
                NotionReporter, PaperReport,
            )
            report = PaperReport(
                paper=paper,
                processing_success=result.extracted,
                processing_duration=result.stage_durations.get("extract", 0),
                codegen_success=result.code_generated,
                codegen_files=result.code_files,
                codegen_output_dir=result.code_dir or None,
                codegen_repo_url=result.repo_url or None,
                codegen_duration=result.stage_durations.get("codegen", 0),
            )
            async with NotionReporter() as notion:
                notion_result = await notion.create_paper_page(report)
                if notion_result and notion_result.get("object") == "page":
                    result.reported = True
                    notion_url = notion_result.get("url", "")
                    report.notion_page_url = notion_url
                    result.notion_url = notion_url
        except Exception as e:
            logger.warning(f"Notion reporting failed: {e}")

        result.stage_durations["report"] = time.time() - t0

    async def run_topic(
        self,
        topic: str,
        *,
        top_n: int = 5,
        force: bool = False,
    ) -> PipelineRunResult:
        """Discover papers on a topic and run the full pipeline.

        Args:
            topic: Search query for paper discovery.
            top_n: Maximum number of papers to process.
            force: If True, re-process even if previously completed.

        Returns:
            A ``PipelineRunResult`` aggregating per-paper outcomes.
        """
        start = time.time()
        run_result = PipelineRunResult(topic=topic)

        self._progress("discover", 0, f"Discovering papers: {topic}")

        from ml_platform.discovery.pipeline import DiscoveryPipeline
        discovery = DiscoveryPipeline()
        discover_result = await discovery.search(
            query=topic, top_n=top_n,
            sources=self.discover_sources,
        )

        run_result.total_papers = len(discover_result.papers)
        self._progress("discover", 5, f"Found {discover_result.total_found} papers")

        # Process each paper with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _run_one(paper: Paper) -> PaperPipelineResult:
            async with semaphore:
                pid = paper.arxiv_id or paper.paper_id.replace("arxiv_", "")
                return await self.run_paper(pid, force=force)

        tasks = [_run_one(p) for p in discover_result.papers]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        self._aggregate_results(raw_results, run_result)

        # Summary Notion page
        if self.report_to_notion:
            await self._create_topic_summary(topic, discover_result, run_result)

        run_result.total_duration = time.time() - start
        return run_result

    def _aggregate_results(
        self,
        raw_results: list,
        run_result: PipelineRunResult,
    ) -> None:
        """Aggregate raw results into the run result counters.

        Args:
            raw_results: List of results or exceptions from gather.
            run_result: The PipelineRunResult to update.
        """
        for r in raw_results:
            if isinstance(r, Exception):
                run_result.failed += 1
                run_result.results.append(PaperPipelineResult(error=str(r)))
            elif r.skipped:
                run_result.skipped += 1
                run_result.results.append(r)
            elif r.success:
                run_result.successful += 1
                run_result.results.append(r)
            else:
                run_result.failed += 1
                run_result.results.append(r)

    async def _create_topic_summary(
        self,
        topic: str,
        discover_result: Any,
        run_result: PipelineRunResult,
    ) -> None:
        """Create a Notion summary page for a topic run.

        Args:
            topic: The search topic query.
            discover_result: The discovery pipeline result.
            run_result: The aggregated pipeline run result.
        """
        try:
            from ml_platform.orchestration.reporter import (
                NotionReporter, PaperReport, PipelineRunSummary,
            )

            paper_reports: list[PaperReport] = []
            for r in run_result.results:
                paper_reports.append(PaperReport(
                    processing_success=r.extracted,
                    codegen_success=r.code_generated,
                    codegen_files=r.code_files,
                    codegen_repo_url=r.repo_url or None,
                ))

            summary = PipelineRunSummary(
                discovery_queries=[topic],
                total_papers_discovered=discover_result.total_found,
                papers_processed=run_result.successful,
                papers_failed=run_result.failed,
                codegen_attempts=sum(1 for r in run_result.results if r.code_generated),
                codegen_successes=sum(1 for r in run_result.results if r.code_generated),
                reports=paper_reports,
            )
            async with NotionReporter() as notion:
                await notion.create_summary_page(summary)
        except Exception as e:
            logger.warning(f"Summary reporting failed: {e}")

    async def run_daily(self, *, top_n: int = 10) -> list[PipelineRunResult]:
        """Run the pipeline for all configured daily topics.

        Args:
            top_n: Maximum number of papers per topic.

        Returns:
            A list of ``PipelineRunResult``, one per topic.
        """
        from ml_platform.config import config

        all_results: list[PipelineRunResult] = []
        for topic in config.DEFAULT_TOPICS:
            self._progress("daily", 0, f"Daily run: {topic}")
            result = await self.run_topic(topic, top_n=top_n)
            all_results.append(result)

        return all_results
