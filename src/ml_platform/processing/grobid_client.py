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
from typing import Any

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
    """Result of a GROBID PDF processing request.

    Attributes:
        tei_xml: TEI XML string returned by GROBID on success.
        success: Whether the processing succeeded.
        error: Error message on failure.
        duration_seconds: Time taken for the GROBID request.
    """

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

    Attributes:
        _base_url: Base URL of the GROBID service.
        _timeout: Per-request timeout in seconds.
        _rate_limit: Minimum seconds between requests.
        _client: The underlying httpx.AsyncClient (lazily created).
        _last_request_time: Timestamp of the last HTTP request for rate limiting.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        rate_limit: float | None = None,
    ) -> None:
        """Initialize the GROBID client.

        Args:
            base_url: Base URL of the GROBID service.  Defaults to
                ``APIConfig.GROBID_BASE_URL`` or ``http://localhost:8070``.
            timeout: Per-request timeout in seconds.  Defaults to 120.0.
            rate_limit: Minimum seconds between requests.  Defaults to 1.0.
        """
        self._base_url = (base_url or getattr(
            APIConfig, "GROBID_BASE_URL", _GROBID_DEFAULT_URL
        )).rstrip("/")
        self._timeout = timeout if timeout is not None else _GROBID_DEFAULT_TIMEOUT
        self._rate_limit = rate_limit if rate_limit is not None else _GROBID_DEFAULT_RATE_LIMIT
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    # --- Context manager ---------------------------------------------------

    async def __aenter__(self) -> GrobidClient:
        """Enter the async context manager and initialise the HTTP client.

        Returns:
            The GrobidClient instance with an active HTTP client.
        """
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit the async context manager and close the HTTP client.

        Args:
            exc_type: Exception type, if an exception was raised.
            exc_val: Exception value, if an exception was raised.
            exc_tb: Exception traceback, if an exception was raised.
        """
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
        """Return the active httpx client, creating a lazy one if needed.

        Returns:
            The active httpx.AsyncClient instance.
        """
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

        Raises:
            httpx.TimeoutException: If the request times out.
            httpx.HTTPStatusError: If an unexpected HTTP error occurs.
            httpx.RequestError: On network-level failures.
        """
        pdf_path = Path(pdf_path)
        validation_error = self._validate_pdf_path(pdf_path)
        if validation_error:
            return validation_error

        start_time = time.monotonic()

        try:
            response = await self._send_grobid_request(pdf_path)
            elapsed = time.monotonic() - start_time

            http_error = self._check_grobid_http_errors(response, elapsed)
            if http_error:
                return http_error

            response.raise_for_status()
            return self._parse_grobid_response(response, pdf_path, elapsed)

        except httpx.TimeoutException as exc:
            return self._grobid_error_result(start_time, f"GROBID request timed out after {time.monotonic() - start_time:.1f}s: {exc}")
        except httpx.HTTPStatusError as exc:
            return self._grobid_error_result(start_time, f"GROBID HTTP {exc.response.status_code}: {exc}")
        except httpx.RequestError as exc:
            return self._grobid_error_result(start_time, f"GROBID network error: {exc}")
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.exception("Unexpected GROBID error")
            return GrobidResult(success=False, error=f"Unexpected error: {exc}", duration_seconds=elapsed)

    @staticmethod
    def _validate_pdf_path(pdf_path: Path) -> GrobidResult | None:
        """Validate that the PDF path exists and is a file.

        Args:
            pdf_path: Path to validate.

        Returns:
            A GrobidResult error if validation fails, or None if valid.
        """
        if not pdf_path.exists():
            return GrobidResult(success=False, error=f"PDF file not found: {pdf_path}")
        if not pdf_path.is_file():
            return GrobidResult(success=False, error=f"Path is not a file: {pdf_path}")
        return None

    async def _send_grobid_request(self, pdf_path: Path) -> httpx.Response:
        """Send the PDF to GROBID for processing.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            The httpx Response from GROBID.
        """
        await self._rate_limit_sleep()
        with open(pdf_path, "rb") as fh:
            files = {"input": (pdf_path.name, fh, "application/pdf")}
            client = self._get_client()
            return await client.post(
                f"{self._base_url}/api/processFulltextDocument",
                files=files,
            )

    @staticmethod
    def _check_grobid_http_errors(response: httpx.Response, elapsed: float) -> GrobidResult | None:
        """Check for GROBID-specific HTTP error status codes.

        Args:
            response: The HTTP response to check.
            elapsed: Elapsed time in seconds.

        Returns:
            A GrobidResult error if an error was found, or None.
        """
        if response.status_code == 503:
            logger.error("GROBID service unavailable (503)")
            return GrobidResult(
                success=False,
                error="GROBID service unavailable (503). The service may be overloaded or starting up.",
                duration_seconds=elapsed,
            )
        if response.status_code == 500:
            logger.error("GROBID internal server error (500)")
            return GrobidResult(
                success=False,
                error=f"GROBID internal server error (500): {response.text[:500]}",
                duration_seconds=elapsed,
            )
        return None

    @staticmethod
    def _parse_grobid_response(response: httpx.Response, pdf_path: Path, elapsed: float) -> GrobidResult:
        """Parse the GROBID response and validate the TEI XML.

        Args:
            response: The HTTP response from GROBID.
            pdf_path: Original PDF path (for logging).
            elapsed: Elapsed time in seconds.

        Returns:
            A GrobidResult with the parsed TEI XML or an error.
        """
        tei_xml = response.text
        if not tei_xml or not tei_xml.strip().startswith("<"):
            logger.error("GROBID returned invalid XML response")
            return GrobidResult(
                success=False,
                error="GROBID returned an invalid or empty XML response.",
                duration_seconds=elapsed,
            )

        logger.info("GROBID processed %s in %.1fs", pdf_path.name, elapsed)
        return GrobidResult(tei_xml=tei_xml, success=True, duration_seconds=elapsed)

    @staticmethod
    def _grobid_error_result(start_time: float, error: str) -> GrobidResult:
        """Create a GrobidResult for an error with duration calculation.

        Args:
            start_time: The monotonic start time.
            error: The error message.

        Returns:
            A GrobidResult indicating failure.
        """
        elapsed = time.monotonic() - start_time
        logger.error(error)
        return GrobidResult(success=False, error=error, duration_seconds=elapsed)
