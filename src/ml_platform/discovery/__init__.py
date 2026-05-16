"""Discovery module — Paper discovery clients and pipeline."""

from ml_platform.discovery.arxiv_client import ArxivClient
from ml_platform.discovery.semantic_scholar_client import SemanticScholarClient
from ml_platform.discovery.paperswithcode_client import PapersWithCodeClient

__all__ = ["ArxivClient", "SemanticScholarClient", "PapersWithCodeClient"]
