"""ML Research Platform — GROBID client for PDF parsing.

Async client for the GROBID REST API (https://github.com/kermitt2/grobid).
Sends PDFs to GROBID for full-text extraction and returns TEI XML.

Usage::

    async with GrobidClient() as client:
        result = await client.process_paper("/path/to/paper.pdf")
        if result.success:
            print(result.tei_xml)
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from ml_platform.config import APIConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

_GROBID_DEFAULT_URL = "http://localhost:8070"
_GROBID_DEFAULT_TIMEOUT = 120.0  # PDF processing can be slow
_GROBID_DEFAULT_RATE_LIMIT = 1.0  # requests per second


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class GrobidResult(BaseModel):
    """Result of a GROBID PDF processing request."""

    tei_xml: str | None = None
    success: bool = False
    error: str | None = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GrobidClient:
    """Async client for the GROBID REST API.

    Supports async context-manager usage for automatic resource cleanup::

        async with GrobidClient() as client:
            alive = await client.check_health()
            result = await client.process_paper("paper.pdf")

    Or manual lifecycle::

        client = GrobidClient()
        try:
            await client.start()
            result = await client.process_paper("paper.pdf")
        finally:
            await client.close()
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        rate_limit: float | None = None,
    ) -> None:
        self._base_url = (base_url or getattr(
            APIConfig, "GROBID_BASE_URL", _GROBID_DEFAULT_URL
        )).rstrip("/")
        self._timeout = timeout if timeout is not None else _GROBID_DEFAULT_TIMEOUT
        self._rate_limit = rate_limit if rate_limit is not None else _GROBID_DEFAULT_RATE_LIMIT
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    # --- Context manager ---------------------------------------------------

    async def __aenter__(self) -> "GrobidClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        await self.close()

    # --- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Initialise the underlying httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                follow_redirects=True,
            )

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- Internal helpers --------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return the active httpx client, creating a lazy one if needed."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                follow_redirects=True,
            )
        return self._client

    async def _rate_limit_sleep(self) -> None:
        """Sleep to respect the configured rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        wait = (1.0 / self._rate_limit) - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_time = time.monotonic()

    # --- Public API --------------------------------------------------------

    async def check_health(self) -> bool:
        """Check whether the GROBID service is alive.

        Returns:
            True if GROBID responds to ``/api/isalive``, False otherwise.
        """
        try:
            client = self._get_client()
            response = await client.get(f"{self._base_url}/api/isalive")
            return response.status_code == 200
        except httpx.RequestError:
            logger.debug("GROBID health check failed (connection error)")
            return False
        except Exception:
            logger.debug("GROBID health check failed (unexpected error)")
            return False

    async def wait_until_healthy(
        self,
        max_retries: int = 30,
        interval: float = 2.0,
    ) -> bool:
        """Block until the GROBID service is healthy or retries are exhausted.

        Args:
            max_retries: Maximum number of health-check attempts.
            interval: Seconds to wait between attempts.

        Returns:
            True if the service became healthy, False otherwise.
        """
        for attempt in range(1, max_retries + 1):
            if await self.check_health():
                logger.info("GROBID service is healthy (attempt %d)", attempt)
                return True
            logger.debug(
                "GROBID not ready, retrying in %.1fs (attempt %d/%d)",
                interval, attempt, max_retries,
            )
            await asyncio.sleep(interval)
        logger.error(
            "GROBID service did not become healthy after %d retries",
            max_retries,
        )
        return False

    async def process_paper(self, pdf_path: str | Path) -> GrobidResult:
        """Send a PDF to GROBID for full-text extraction.

        Posts the PDF to ``/api/processFulltextDocument`` and returns the
        TEI XML response.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            A :class:`GrobidResult` with ``tei_xml`` populated on success,
            or ``error`` populated on failure.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return GrobidResult(
                success=False,
                error=f"PDF file not found: {pdf_path}",
            )
        if not pdf_path.is_file():
            return GrobidResult(
                success=False,
                error=f"Path is not a file: {pdf_path}",
            )

        start_time = time.monotonic()

        try:
            await self._rate_limit_sleep()

            with open(pdf_path, "rb") as fh:
                files = {"input": (pdf_path.name, fh, "application/pdf")}

                client = self._get_client()
                response = await client.post(
                    f"{self._base_url}/api/processFulltextDocument",
                    files=files,
                )

            elapsed = time.monotonic() - start_time

            # --- Handle GROBID-specific HTTP errors ---
            if response.status_code == 503:
                logger.error("GROBID service unavailable (503)")
                return GrobidResult(
                    success=False,
                    error="GROBID service unavailable (503). "
                          "The service may be overloaded or starting up.",
                    duration_seconds=elapsed,
                )

            if response.status_code == 500:
                logger.error("GROBID internal server error (500)")
                return GrobidResult(
                    success=False,
                    error=f"GROBID internal server error (500): "
                          f"{response.text[:500]}",
                    duration_seconds=elapsed,
                )

            response.raise_for_status()

            tei_xml = response.text
            if not tei_xml or not tei_xml.strip().startswith("<"):
                logger.error("GROBID returned invalid XML response")
                return GrobidResult(
                    success=False,
                    error="GROBID returned an invalid or empty XML response.",
                    duration_seconds=elapsed,
                )

            logger.info(
                "GROBID processed %s in %.1fs", pdf_path.name, elapsed,
            )
            return GrobidResult(
                tei_xml=tei_xml,
                success=True,
                duration_seconds=elapsed,
            )

        except httpx.TimeoutException as exc:
            elapsed = time.monotonic() - start_time
            logger.error("GROBID request timed out: %s", exc)
            return GrobidResult(
                success=False,
                error=f"GROBID request timed out after {elapsed:.1f}s: {exc}",
                duration_seconds=elapsed,
            )

        except httpx.HTTPStatusError as exc:
            elapsed = time.monotonic() - start_time
            logger.error("GROBID HTTP error: %s", exc)
            return GrobidResult(
                success=False,
                error=f"GROBID HTTP {exc.response.status_code}: {exc}",
                duration_seconds=elapsed,
            )

        except httpx.RequestError as exc:
            elapsed = time.monotonic() - start_time
            logger.error("GROBID network error: %s", exc)
            return GrobidResult(
                success=False,
                error=f"GROBID network error: {exc}",
                duration_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.exception("Unexpected GROBID error")
            return GrobidResult(
                success=False,
                error=f"Unexpected error: {exc}",
                duration_seconds=elapsed,
            )
