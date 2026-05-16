"""Processing module — PDF download, GROBID parsing, metadata enrichment."""

from ml_platform.processing.pdf_downloader import PDFDownloader, DownloadResult
from ml_platform.processing.grobid_client import GrobidClient, GrobidResult
from ml_platform.processing.tei_parser import parse_tei_xml, update_paper, TeiParseResult

__all__ = [
    "PDFDownloader",
    "DownloadResult",
    "GrobidClient",
    "GrobidResult",
    "parse_tei_xml",
    "update_paper",
    "TeiParseResult",
]
