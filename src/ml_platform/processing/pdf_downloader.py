"""ML Research Platform — Async PDF downloader.

Downloads and caches PDFs from arXiv and open-access sources (e.g. Semantic
Scholar).  Uses httpx for async HTTP, respects per-domain rate limits, and
stores files under ``data/pdfs/`` keyed by arXiv ID or a sanitized filename.

Usage::

    async with PDFDownloader() as downloader:
        result = await downloader.download_paper(paper)
        if result.success:
            print(f"Saved to {result.path}")
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx
from pydantic import BaseModel, Field

from ml_platform.config import APIConfig, AppConfig
from ml_platform.models import Paper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class DownloadResult(BaseModel):
    """Outcome of a single PDF download attempt."""

    paper_id: str = Field(description="Paper.paper_id that was requested")
    path: Path | None = Field(default=None, description="Local file path on success")
    size_bytes: int = Field(default=0, description="File size in bytes (0 on failure)")
    success: bool = Field(default=False)
    skipped: bool = Field(default=False, description="True when file already existed")
    error: str | None = Field(default=None, description="Error message on failure")
    duration_seconds: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal PDF signature: first bytes should be ``%PDF``
_PDF_MAGIC = b"%PDF"

_ARXIV_PDF_RE = re.compile(r"^https?://arxiv\.org/pdf/")

def _sanitize_filename(name: str) -> str:
    """Return a filesystem-safe filename stem from an arbitrary string."""
    # Replace anything that isn't alphanumeric, dash, dot, or underscore
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _resolve_local_path(paper: Paper, pdf_dir: Path) -> Path:
    """Determine the local filename for a paper's PDF.

    Strategy:
      1. If ``paper.arxiv_id`` is set → ``{arxiv_id}.pdf``
      2. Else sanitise ``paper.paper_id`` → ``{sanitized}.pdf``
    """
    stem = paper.arxiv_id if paper.arxiv_id else _sanitize_filename(paper.paper_id)
    return pdf_dir / f"{stem}.pdf"


def _resolve_pdf_url(paper: Paper) -> str | None:
    """Return the best PDF URL for *paper*, or None if unavailable.

    Prefers arXiv direct links; falls back to ``paper.pdf_url`` (which may
    come from Semantic Scholar's ``openAccessPdf``).
    """
    # Explicit arXiv ID always wins
    if paper.arxiv_id:
        return f"https://arxiv.org/pdf/{paper.arxiv_id}"
    # If pdf_url is already an arXiv PDF link, use it as-is
    if paper.pdf_url and _ARXIV_PDF_RE.match(paper.pdf_url):
        return paper.pdf_url
    # Generic open-access URL (Semantic Scholar, Unpaywall, etc.)
    if paper.pdf_url:
        return paper.pdf_url
    return None


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

# Type alias for the optional progress callback
ProgressCallback = Callable[[str, int, int], None]  # (paper_id, bytes_downloaded, total_bytes)


class PDFDownloader:
    """Async PDF downloader with rate-limiting and caching.

    Usage::

        async with PDFDownloader() as dl:
            result = await dl.download_paper(paper)

    Parameters
    ----------
    pdf_dir:
        Directory to store downloaded PDFs.  Defaults to ``AppConfig.PDF_DIR``.
    rate_limit:
        Minimum seconds between requests to the same domain.
        Defaults to ``APIConfig.ARXIV_RATE_LIMIT`` (1 req/s).
    timeout:
        Per-request timeout in seconds.
    progress_callback:
        Optional callable ``f(paper_id, bytes_downloaded, total_bytes)``
        invoked during streaming downloads to report progress.
    """

    def __init__(
        self,
        pdf_dir: Path | None = None,
        rate_limit: float | None = None,
        timeout: float = 60.0,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._pdf_dir = pdf_dir or AppConfig.PDF_DIR
        self._rate_limit = rate_limit if rate_limit is not None else APIConfig.ARXIV_RATE_LIMIT
        self._timeout = timeout
        self._progress_callback = progress_callback
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> PDFDownloader:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=10.0),
            follow_redirects=True,
            headers={"User-Agent": "ml-research-platform/0.1 (mailto:research@example.com)"},
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the active httpx client, lazily creating one if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                follow_redirects=True,
                headers={"User-Agent": "ml-research-platform/0.1 (mailto:research@example.com)"},
            )
        return self._client

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _rate_limit_sleep(self) -> None:
        """Sleep to respect the configured rate limit."""
        if self._rate_limit <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_time
        wait = self._rate_limit - elapsed
        if wait > 0:
            logger.debug("Rate-limit: sleeping %.2fs", wait)
            await asyncio.sleep(wait)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_pdf(data: bytes) -> bool:
        """Heuristic: check that the first bytes match ``%PDF``."""
        return data[:4] == _PDF_MAGIC

    # ------------------------------------------------------------------
    # Core download
    # ------------------------------------------------------------------

    async def _download_bytes(
        self,
        url: str,
        paper_id: str,
    ) -> tuple[bytes, int]:
        """Stream-download *url* and return ``(raw_bytes, status_code)``.

        Raises on network-level failures so callers can decide what to do.
        """
        await self._rate_limit_sleep()
        client = self._get_client()

        logger.info("Downloading PDF for %s from %s", paper_id, url)

        chunks: list[bytes] = []
        total_bytes = 0

        async with client.stream("GET", url) as response:
            if response.status_code == 404:
                return b"", 404
            response.raise_for_status()

            # Check Content-Type hint (warn only, don't block)
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type and "octet-stream" not in content_type:
                logger.warning(
                    "Unexpected Content-Type for %s: %s — proceeding anyway",
                    paper_id,
                    content_type,
                )

            total_hint = int(response.headers.get("content-length", "0"))

            async for chunk in response.aiter_bytes(chunk_size=65_536):
                chunks.append(chunk)
                total_bytes += len(chunk)
                if self._progress_callback:
                    try:
                        self._progress_callback(paper_id, total_bytes, total_hint)
                    except Exception:
                        # Don't let a broken callback kill the download
                        pass

        return b"".join(chunks), 200

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def download_paper(
        self,
        paper: Paper,
        *,
        force: bool = False,
    ) -> DownloadResult:
        """Download the PDF for *paper* and cache it locally.

        Parameters
        ----------
        paper:
            A :class:`Paper` with ``arxiv_id`` or ``pdf_url`` set.
        force:
            If *True*, re-download even if a local file already exists.

        Returns:
            A :class:`DownloadResult` indicating success/failure.
        """
        t0 = time.monotonic()

        pdf_url = _resolve_pdf_url(paper)
        if pdf_url is None:
            return DownloadResult(
                paper_id=paper.paper_id,
                success=False,
                error="No PDF URL available (missing arxiv_id and pdf_url)",
                duration_seconds=time.monotonic() - t0,
            )

        local_path = _resolve_local_path(paper, self._pdf_dir)

        # Ensure output directory exists
        self._pdf_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded
        if not force and local_path.exists() and local_path.stat().st_size > 0:
            existing_size = local_path.stat().st_size
            logger.debug("PDF already cached: %s (%d bytes)", local_path, existing_size)
            return DownloadResult(
                paper_id=paper.paper_id,
                path=local_path,
                size_bytes=existing_size,
                success=True,
                skipped=True,
                duration_seconds=time.monotonic() - t0,
            )

        # Download
        try:
            data, status_code = await self._download_bytes(pdf_url, paper.paper_id)
        except httpx.TimeoutException as exc:
            msg = f"Timeout downloading PDF: {exc}"
            logger.error(msg)
            return DownloadResult(
                paper_id=paper.paper_id,
                success=False,
                error=msg,
                duration_seconds=time.monotonic() - t0,
            )
        except httpx.HTTPStatusError as exc:
            msg = f"HTTP {exc.response.status_code} for {pdf_url}: {exc}"
            logger.error(msg)
            return DownloadResult(
                paper_id=paper.paper_id,
                success=False,
                error=msg,
                duration_seconds=time.monotonic() - t0,
            )
        except httpx.RequestError as exc:
            msg = f"Network error downloading PDF: {exc}"
            logger.error(msg)
            return DownloadResult(
                paper_id=paper.paper_id,
                success=False,
                error=msg,
                duration_seconds=time.monotonic() - t0,
            )

        # Handle 404 returned as status (not raised)
        if status_code == 404 or len(data) == 0:
            msg = f"PDF not found (HTTP 404) at {pdf_url}"
            logger.warning(msg)
            return DownloadResult(
                paper_id=paper.paper_id,
                success=False,
                error=msg,
                duration_seconds=time.monotonic() - t0,
            )

        # Validate it looks like a PDF
        if not self._looks_like_pdf(data):
            msg = (
                f"Downloaded content does not appear to be a valid PDF "
                f"(first bytes: {data[:20]!r})"
            )
            logger.error(msg)
            return DownloadResult(
                paper_id=paper.paper_id,
                success=False,
                error=msg,
                duration_seconds=time.monotonic() - t0,
            )

        # Write to disk
        try:
            local_path.write_bytes(data)
        except OSError as exc:
            msg = f"Failed to write PDF to {local_path}: {exc}"
            logger.error(msg)
            return DownloadResult(
                paper_id=paper.paper_id,
                success=False,
                error=msg,
                duration_seconds=time.monotonic() - t0,
            )

        size = len(data)
        logger.info(
            "Saved PDF for %s → %s (%d bytes)", paper.paper_id, local_path, size
        )
        return DownloadResult(
            paper_id=paper.paper_id,
            path=local_path,
            size_bytes=size,
            success=True,
            duration_seconds=time.monotonic() - t0,
        )

    async def download_papers(
        self,
        papers: list[Paper],
        *,
        force: bool = False,
    ) -> list[DownloadResult]:
        """Download PDFs for multiple papers sequentially (rate-limited).

        Parameters
        ----------
        papers:
            List of :class:`Paper` objects.
        force:
            If *True*, re-download even if a local file already exists.

        Returns:
            A list of :class:`DownloadResult` in the same order as *papers*.
        """
        results: list[DownloadResult] = []
        for i, paper in enumerate(papers):
            logger.info("Downloading %d/%d: %s", i + 1, len(papers), paper.paper_id)
            result = await self.download_paper(paper, force=force)
            results.append(result)
        return results

    async def download_papers_concurrent(
        self,
        papers: list[Paper],
        *,
        force: bool = False,
        max_concurrency: int = 3,
    ) -> list[DownloadResult]:
        """Download PDFs for multiple papers concurrently.

        Uses a semaphore to limit concurrency while still respecting the
        per-request rate limit.

        Parameters
        ----------
        papers:
            List of :class:`Paper` objects.
        force:
            If *True*, re-download even if a local file already exists.
        max_concurrency:
            Maximum number of simultaneous downloads.

        Returns:
            A list of :class:`DownloadResult` in the same order as *papers*.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _guarded(paper: Paper) -> DownloadResult:
            async with semaphore:
                return await self.download_paper(paper, force=force)

        return list(await asyncio.gather(*[_guarded(p) for p in papers]))
