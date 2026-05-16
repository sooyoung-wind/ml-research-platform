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
from typing import Callable, Optional

from ml_platform.models import Paper

logger = logging.getLogger(__name__)


@dataclass
class PaperPipelineResult:
    """Result of running the full pipeline on a single paper."""

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
    """Result of running the pipeline on multiple papers."""

    total_papers: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[PaperPipelineResult] = field(default_factory=list)
    total_duration: float = 0.0
    topic: str = ""


class ResearchOrchestrator:
    """Full pipeline orchestrator for ML research automation."""

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
        if self.progress_callback:
            self.progress_callback(stage, pct, msg)
        logger.info(f"[{stage}] {pct:.0f}% — {msg}")

    async def run_paper(
        self,
        arxiv_id: str,
        *,
        force: bool = False,
    ) -> PaperPipelineResult:
        """Run the full pipeline on a single paper by arXiv ID."""
        start = time.time()
        result = PaperPipelineResult(paper_id=arxiv_id)

        try:
            # ── Stage 1: Create Paper & Download ───────────────────────
            self._progress("download", 10, f"Downloading PDF for {arxiv_id}")
            t0 = time.time()

            from ml_platform.models import PaperSource
            paper = Paper(
                paper_id=f"arxiv_{arxiv_id}",
                arxiv_id=arxiv_id,
                title=arxiv_id,
                source=PaperSource.ARXIV,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )

            from ml_platform.processing.pdf_downloader import PDFDownloader
            async with PDFDownloader() as dl:
                dl_result = await dl.download_paper(paper, force=force)

            if dl_result.success and dl_result.path:
                result.downloaded = True
                result.pdf_path = str(dl_result.path)
                paper.local_pdf_path = result.pdf_path
            else:
                result.error = f"PDF download failed: {dl_result.error}"
                result.total_duration = time.time() - start
                return result

            result.stage_durations["download"] = time.time() - t0

            # ── Stage 2: Extract Text & Enrich ────────────────────────
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

            # ── Stage 3: Code Generation ──────────────────────────────
            if self.skip_codegen:
                self._progress("codegen", 50, "Skipping code generation")
            else:
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

            # ── Stage 4: GitHub Push ──────────────────────────────────
            if result.code_generated and self.push_to_github:
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

            # ── Stage 5: Notion Report ────────────────────────────────
            if self.report_to_notion:
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

            result.success = True

        except Exception as e:
            result.error = str(e)
            logger.error(f"Pipeline failed for {arxiv_id}: {e}", exc_info=True)

        result.total_duration = time.time() - start
        return result

    async def run_topic(
        self,
        topic: str,
        *,
        top_n: int = 5,
        force: bool = False,
    ) -> PipelineRunResult:
        """Discover papers on a topic and run the full pipeline."""
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

        # Summary Notion page
        if self.report_to_notion:
            try:
                from ml_platform.orchestration.reporter import (
                    NotionReporter, PaperReport, PipelineRunSummary,
                )
                from datetime import datetime, timezone

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

        run_result.total_duration = time.time() - start
        return run_result

    async def run_daily(self, *, top_n: int = 10) -> list[PipelineRunResult]:
        """Run the pipeline for all configured daily topics."""
        from ml_platform.config import config

        all_results: list[PipelineRunResult] = []
        for topic in config.DEFAULT_TOPICS:
            self._progress("daily", 0, f"Daily run: {topic}")
            result = await self.run_topic(topic, top_n=top_n)
            all_results.append(result)

        return all_results
