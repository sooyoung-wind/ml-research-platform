"""Discovery module — Paper discovery clients and pipeline."""

from ml_platform.discovery.arxiv_client import ArxivClient
from ml_platform.discovery.huggingface_client import HuggingFaceClient
from ml_platform.discovery.semantic_scholar_client import SemanticScholarClient

__all__ = ["ArxivClient", "SemanticScholarClient", "HuggingFaceClient"]
