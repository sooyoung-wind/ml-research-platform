"""ML Research Platform — Simplified paper processor.

Phase 3 redesign: GROBID is now optional (kept as fallback for PaperCoder).
Default pipeline uses PyPDF2 for text extraction (same as DeepCode).

Pipeline:
  Paper → PDF Download → Text Extract (PyPDF2) → Metadata Enrich → Store
                                        ↘ (optional) GROBID → TEI Parse
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from ml_platform.models import Paper

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single paper.

    Attributes:
        paper_id: Identifier of the processed paper.
        success: Whether processing completed successfully.
        error: Error message if processing failed.
        download: Download stage result dictionary.
        extracted: Text extraction result dictionary.
        enriched: Whether metadata enrichment succeeded.
        duration: Total processing duration in seconds.
    """

    paper_id: str = ""
    success: bool = False
    error: str = ""
    download: dict = field(default_factory=dict)
    extracted: dict = field(default_factory=dict)  # text extraction result
    enriched: bool = False
    duration: float = 0.0


class PaperProcessor:
    """Simplified paper processor — PDF download + text extraction + enrichment.

    No longer requires GROBID by default. Uses PyPDF2 for text extraction,
    matching DeepCode's approach. GROBID remains available as an optional
    advanced parser.

    Attributes:
        use_grobid: Whether to use GROBID for structured parsing.
        enrich_metadata: Whether to enrich papers with external metadata.
    """

    def __init__(
        self,
        *,
        use_grobid: bool = False,
        enrich_metadata: bool = True,
    ) -> None:
        """Initialize the PaperProcessor.

        Args:
            use_grobid: If True, use GROBID for structured PDF parsing
                instead of PyPDF2.
            enrich_metadata: If True, enrich papers with metadata from
                external sources (S2, HuggingFace).
        """
        self.use_grobid = use_grobid
        self.enrich_metadata = enrich_metadata

    async def process_paper(
        self,
        paper: Paper,
        *,
        download: bool = True,
        extract: bool = True,
        enrich: bool = True,
        force: bool = False,
    ) -> ProcessingResult:
        """Process a single paper through the pipeline.

        Args:
            paper: The Paper object to process.
            download: Whether to download the PDF.
            extract: Whether to extract text from the PDF.
            enrich: Whether to enrich with external metadata.
            force: If True, re-download and re-process even if cached.

        Returns:
            A ProcessingResult with details of each processing stage.
        """
        import asyncio
        import time

        start = time.time()
        result = ProcessingResult(paper_id=paper.paper_id)

        try:
            # Step 1: Download PDF
            if download and paper.pdf_url:
                result.download = await self._download_pdf(paper, force=force)

            # Step 2: Extract text
            if extract and paper.local_pdf_path:
                if self.use_grobid:
                    # GROBID path (optional, more structured)
                    result.extracted = await self._extract_grobid(paper)
                else:
                    # PyPDF2 path (default, fast)
                    result.extracted = self._extract_text(paper)

            # Step 3: Enrich metadata
            if enrich and self.enrich_metadata:
                result.enriched = await self._enrich(paper)

            result.success = True

        except Exception as e:
            result.error = str(e)
            logger.error(f"Processing failed for {paper.paper_id}: {e}")

        result.duration = time.time() - start
        return result

    async def process_batch(
        self,
        papers: list[Paper],
        *,
        download: bool = True,
        extract: bool = True,
        enrich: bool = True,
        force: bool = False,
        max_concurrent: int = 3,
    ) -> list[ProcessingResult]:
        """Process multiple papers with concurrency control.

        Args:
            papers: List of Paper objects to process.
            download: Whether to download PDFs.
            extract: Whether to extract text.
            enrich: Whether to enrich with metadata.
            force: If True, re-process even if already done.
            max_concurrent: Maximum number of concurrent processing tasks.

        Returns:
            List of ProcessingResult objects, one per paper.
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _process(paper: Paper) -> ProcessingResult:
            async with semaphore:
                return await self.process_paper(
                    paper,
                    download=download,
                    extract=extract,
                    enrich=enrich,
                    force=force,
                )

        tasks = [_process(p) for p in papers]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[ProcessingResult] = []
        for r in raw:
            if isinstance(r, Exception):
                results.append(ProcessingResult(error=str(r)))
            else:
                results.append(r)  # type: ignore[arg-type]
        return results

    # ── Internal methods ──────────────────────────────────────────────────

    async def _download_pdf(self, paper: Paper, *, force: bool = False) -> dict:
        """Download PDF for a paper.

        Args:
            paper: The Paper to download.
            force: If True, re-download even if file exists.

        Returns:
            A dictionary with download status and file information.
        """
        from ml_platform.processing.pdf_downloader import PDFDownloader

        async with PDFDownloader() as downloader:
            result = await downloader.download_paper(paper, force=force)

        if result.success and result.path:
            paper.local_pdf_path = str(result.path)
            return {
                "success": True,
                "path": str(result.path),
                "size_bytes": result.size_bytes,
            }
        return {"success": False, "error": result.error}

    def _extract_text(self, paper: Paper) -> dict:
        """Extract text from PDF using PyPDF2 (fast, no GROBID needed).

        Args:
            paper: The Paper with a local_pdf_path set.

        Returns:
            A dictionary with extraction status, page count, and character
            count.
        """
        from PyPDF2 import PdfReader

        if not paper.local_pdf_path or not os.path.exists(paper.local_pdf_path):
            return {"success": False, "error": "No PDF file available"}

        try:
            reader = PdfReader(paper.local_pdf_path)
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append(text)

            full_text = "\n\n".join(pages)
            paper.abstract = paper.abstract or ""

            # Store extracted text in parsed_content
            paper.parsed_content = {
                "full_text": full_text,
                "pages": len(reader.pages),
                "extraction_method": "pypdf2",
            }

            return {
                "success": True,
                "pages": len(reader.pages),
                "chars": len(full_text),
                "extraction_method": "pypdf2",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _extract_grobid(self, paper: Paper) -> dict:
        """Extract structured content using GROBID (optional, more detailed).

        Args:
            paper: The Paper with a local_pdf_path set.

        Returns:
            A dictionary with structured extraction results including
            sections, references, figures, and keywords counts.
        """
        from ml_platform.processing.grobid_client import GrobidClient
        from ml_platform.processing.tei_parser import parse_tei_xml, update_paper

        if not paper.local_pdf_path or not os.path.exists(paper.local_pdf_path):
            return {"success": False, "error": "No PDF file available"}

        async with GrobidClient() as grobid:
            grobid_result = await grobid.process_paper(paper.local_pdf_path)

        if not grobid_result.success or not grobid_result.tei_xml:
            return {"success": False, "error": grobid_result.error}

        parse_result = parse_tei_xml(grobid_result.tei_xml)
        if parse_result.title:
            paper.title = parse_result.title
        if parse_result.abstract:
            paper.abstract = parse_result.abstract

        errors = update_paper(paper, parse_result)

        return {
            "success": True,
            "sections": len(parse_result.sections),
            "references": len(parse_result.references),
            "figures": len(parse_result.figures),
            "keywords": len(parse_result.keywords),
            "extraction_method": "grobid",
            "errors": errors if errors else None,
        }

    async def _enrich(self, paper: Paper) -> bool:
        """Enrich paper with metadata from S2 and HuggingFace.

        Args:
            paper: The Paper to enrich.

        Returns:
            True if enrichment succeeded, False otherwise.
        """
        try:
            from ml_platform.processing.enricher import MetadataEnricher

            async with MetadataEnricher() as enricher:  # type: ignore[attr-defined]
                await enricher.enrich(paper)
            return True
        except Exception as e:
            logger.warning(f"Enrichment failed for {paper.paper_id}: {e}")
            return False
