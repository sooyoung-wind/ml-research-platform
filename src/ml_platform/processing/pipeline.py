"""ML Research Platform — Processing pipeline orchestrator.

Coordinates the full processing chain:
  Paper → PDF Download → GROBID Parse → TEI Extract → Enrich → Store
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ml_platform.db import PapersDB
from ml_platform.models import Paper, ProcessingStatus
from ml_platform.processing.enricher import MetadataEnricher
from ml_platform.processing.grobid_client import GrobidClient
from ml_platform.processing.pdf_downloader import PDFDownloader
from ml_platform.processing.tei_parser import parse_tei_xml, update_paper

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Orchestrates paper processing: download, parse, enrich, store."""

    def __init__(self) -> None:
        self.db = PapersDB()
        self.downloader = PDFDownloader()
        self.grobid = GrobidClient()
        self.enricher = MetadataEnricher()

    async def process_paper(
        self,
        paper: Paper,
        *,
        download: bool = True,
        parse: bool = True,
        enrich: bool = True,
        force: bool = False,
    ) -> ProcessingResult:
        """Process a single paper through the full pipeline.

        Args:
            paper: Paper to process.
            download: Whether to download PDF.
            parse: Whether to parse with GROBID.
            enrich: Whether to enrich with external metadata.
            force: Re-download/re-process even if already done.

        Returns:
            ProcessingResult with details of each stage.
        """
        result = ProcessingResult(paper_id=paper.paper_id)
        start = time.time()

        # Stage 1: Download PDF
        if download:
            dl_result = await self._download(paper, force=force)
            result.download = dl_result
            if not dl_result["success"] and not dl_result.get("skipped"):
                result.success = False
                result.error = f"Download failed: {dl_result.get('error')}"
                result.duration = round(time.time() - start, 2)
                return result

        # Stage 2: Parse with GROBID
        if parse and paper.local_pdf_path:
            parse_result = await self._parse(paper, force=force)
            result.parsed = parse_result
            if not parse_result["success"]:
                logger.warning("Parse failed for %s: %s", paper.paper_id, parse_result.get("error"))

        # Stage 3: Enrich metadata
        if enrich:
            try:
                paper = await self.enricher.enrich(paper)
                result.enriched = True
            except Exception as exc:
                logger.warning("Enrichment failed for %s: %s", paper.paper_id, exc)
                result.enriched = False

        # Stage 4: Store
        paper.status = ProcessingStatus.PARSED if paper.parsed_content else ProcessingStatus.PDF_DOWNLOADED
        paper.updated_at = __import__("datetime").datetime.now()
        self.db.upsert_papers([paper])
        result.stored = True

        result.success = True
        result.duration = round(time.time() - start, 2)
        return result

    async def process_batch(
        self,
        papers: list[Paper],
        *,
        download: bool = True,
        parse: bool = True,
        enrich: bool = True,
        force: bool = False,
        max_concurrent: int = 3,
    ) -> list[ProcessingResult]:
        """Process multiple papers with concurrency control.

        Args:
            papers: List of papers to process.
            max_concurrent: Max parallel downloads/parses.

        Returns:
            List of ProcessingResults.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[ProcessingResult] = []

        async def _process_one(paper: Paper) -> ProcessingResult:
            async with semaphore:
                return await self.process_paper(
                    paper,
                    download=download,
                    parse=parse,
                    enrich=enrich,
                    force=force,
                )

        tasks = [_process_one(p) for p in papers]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unwrap exceptions
        final: list[ProcessingResult] = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                final.append(ProcessingResult(
                    paper_id=papers[i].paper_id,
                    success=False,
                    error=str(r),
                ))
            else:
                final.append(r)  # type: ignore[arg-type]

        logger.info(
            "Processed %d papers: %d success, %d failed",
            len(final),
            sum(1 for r in final if r.success),
            sum(1 for r in final if not r.success),
        )
        return final

    # --- Internal stages ---------------------------------------------------

    async def _download(self, paper: Paper, *, force: bool = False) -> dict:
        """Download PDF for a paper."""
        async with self.downloader:
            result = await self.downloader.download_paper(paper, force=force)

        if result.success and result.path:
            paper.local_pdf_path = str(result.path)
            paper.status = ProcessingStatus.PDF_DOWNLOADED

        return {
            "success": result.success,
            "skipped": result.skipped,
            "path": str(result.path) if result.path else None,
            "size_bytes": result.size_bytes,
            "error": result.error,
        }

    async def _parse(self, paper: Paper, *, force: bool = False) -> dict:
        """Parse a paper's PDF with GROBID and extract structured content."""
        pdf_path = Path(paper.local_pdf_path) if paper.local_pdf_path else None
        if not pdf_path or not pdf_path.exists():
            return {"success": False, "error": "PDF file not found"}

        # Skip if already parsed
        if paper.parsed_content and not force:
            return {"success": True, "skipped": True, "sections": len(paper.parsed_content.get("sections", []))}

        async with self.grobid:
            grobid_result = await self.grobid.process_paper(str(pdf_path))

        if not grobid_result.success or not grobid_result.tei_xml:
            return {"success": False, "error": grobid_result.error}

        # Parse TEI XML
        parse_result = parse_tei_xml(grobid_result.tei_xml)

        # Update paper model
        paper = update_paper(paper, parse_result)
        paper.status = ProcessingStatus.PARSED

        return {
            "success": True,
            "skipped": False,
            "sections": len(parse_result.sections),
            "references": len(parse_result.references),
            "figures": len(parse_result.figures),
            "partial": parse_result.partial,
            "errors": parse_result.parse_errors,
        }


class ProcessingResult:
    """Result of processing a single paper."""

    def __init__(
        self,
        paper_id: str,
        success: bool = False,
        error: str | None = None,
        download: dict | None = None,
        parsed: dict | None = None,
        enriched: bool = False,
        stored: bool = False,
        duration: float = 0.0,
    ) -> None:
        self.paper_id = paper_id
        self.success = success
        self.error = error
        self.download = download
        self.parsed = parsed
        self.enriched = enriched
        self.stored = stored
        self.duration = duration

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAIL({self.error})"
        return f"ProcessingResult({self.paper_id}: {status}, {self.duration:.1f}s)"
