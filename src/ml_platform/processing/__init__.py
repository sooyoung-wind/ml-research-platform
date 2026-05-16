"""Processing module — PDF download, text extraction, metadata enrichment.

Default pipeline (PyPDF2, no GROBID required):
  Paper → PDF Download → PyPDF2 Text Extract → Enrich → Store

Optional GROBID pipeline (more structured):
  Paper → PDF Download → GROBID → TEI Parse → Enrich → Store
"""

from ml_platform.processing.pdf_downloader import PDFDownloader, DownloadResult
from ml_platform.processing.processor import PaperProcessor, ProcessingResult
from ml_platform.processing.enricher import MetadataEnricher

# Optional — only available when GROBID Docker is running
try:
    from ml_platform.processing.grobid_client import GrobidClient, GrobidResult
    from ml_platform.processing.tei_parser import parse_tei_xml, update_paper, TeiParseResult
except ImportError:
    pass

__all__ = [
    "PDFDownloader",
    "DownloadResult",
    "PaperProcessor",
    "ProcessingResult",
    "MetadataEnricher",
]
