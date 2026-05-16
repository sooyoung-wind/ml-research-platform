"""ML Research Platform — Configuration management.

Provides centralized configuration for API keys, rate limits,
directory paths, and default settings used across the platform.

Attributes:
    PROJECT_ROOT: Path to the project root directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class APIConfig:
    """API keys and rate limit settings for external services.

    Attributes:
        SEMANTIC_SCHOLAR_API_KEY: API key for Semantic Scholar.
        SEMANTIC_SCHOLAR_BASE_URL: Base URL for Semantic Scholar API.
        SEMANTIC_SCHOLAR_RATE_LIMIT: Requests per second (with key: 10).
        ARXIV_BASE_URL: Base URL for arXiv API.
        ARXIV_RATE_LIMIT: Requests per second for arXiv.
        HUGGINGFACE_PAPERS_BASE_URL: Base URL for HuggingFace Papers.
        HUGGINGFACE_PAPERS_RATE_LIMIT: Requests per second for HuggingFace.
        OPENALEX_BASE_URL: Base URL for OpenAlex API.
        OPENALEX_MAILTO: Email for OpenAlex polite pool.
        CROSSREF_BASE_URL: Base URL for CrossRef API.
        CROSSREF_MAILTO: Email for CrossRef polite pool.
        CORE_API_KEY: API key for CORE.
        CORE_BASE_URL: Base URL for CORE API.
        UNPAYWALL_EMAIL: Email for Unpaywall API.
        GROBID_BASE_URL: Base URL for GROBID service.
        GROBID_RATE_LIMIT: Requests per second for GROBID.
        GROBID_TIMEOUT: Timeout in seconds for GROBID PDF processing.
        OPENAI_API_KEY: API key for OpenAI.
        OPENAI_MODEL: Default OpenAI model name.
        ANTHROPIC_API_KEY: API key for Anthropic.
        GOOGLE_API_KEY: API key for Google/Gemini.
    """

    # Semantic Scholar
    SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    SEMANTIC_SCHOLAR_BASE_URL: str = "https://api.semanticscholar.org/graph/v1"
    SEMANTIC_SCHOLAR_RATE_LIMIT: float = 1.0  # requests per second (with key: 10)

    # arXiv
    ARXIV_BASE_URL: str = "https://export.arxiv.org/api/query"
    ARXIV_RATE_LIMIT: float = 1.0  # generous, no auth needed

    # HuggingFace Papers (replaces defunct PapersWithCode)
    HUGGINGFACE_PAPERS_BASE_URL: str = "https://huggingface.co"
    HUGGINGFACE_PAPERS_RATE_LIMIT: float = 2.0  # no auth needed

    # OpenAlex
    OPENALEX_BASE_URL: str = "https://api.openalex.org"
    OPENALEX_MAILTO: str = os.getenv("OPENALEX_MAILTO", "")

    # CrossRef
    CROSSREF_BASE_URL: str = "https://api.crossref.org"
    CROSSREF_MAILTO: str = os.getenv("CROSSREF_MAILTO", "")

    # CORE
    CORE_API_KEY: str = os.getenv("CORE_API_KEY", "")
    CORE_BASE_URL: str = "https://api.core.ac.uk/v3"

    # Unpaywall
    UNPAYWALL_EMAIL: str = os.getenv("UNPAYWALL_EMAIL", "")

    # GROBID (PDF parsing)
    GROBID_BASE_URL: str = os.getenv("GROBID_BASE_URL", "http://localhost:8070")
    GROBID_RATE_LIMIT: float = 1.0  # requests per second
    GROBID_TIMEOUT: float = 120.0  # PDF processing can be slow

    # OpenAI (code generation)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "o3-mini")

    # Anthropic (code generation)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Google (code generation)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))


class AppConfig:
    """Application-level settings for directories, defaults, and weights.

    Attributes:
        PROJECT_ROOT: Path to the project root directory.
        DATA_DIR: Path to the data directory.
        PDF_DIR: Path to the PDF storage directory.
        GENERATED_DIR: Path to the generated code output directory.
        DB_PATH: Path to the SQLite database file.
        DEFAULT_TOPICS: List of default research topics for discovery.
        DEFAULT_ARXIV_CATEGORIES: Default arXiv categories for paper search.
        DEFAULT_TOP_N: Default number of papers to retrieve.
        WEIGHT_CITATIONS: Weight for citation count in ranking.
        WEIGHT_RELEVANCE: Weight for relevance score in ranking.
        WEIGHT_FRESHNESS: Weight for freshness in ranking.
        WEIGHT_NO_CODE: Weight bonus for papers without code implementation.
    """

    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    PDF_DIR: Path = DATA_DIR / "pdfs"
    GENERATED_DIR: Path = DATA_DIR / "generated"
    DB_PATH: Path = DATA_DIR / "papers.db"

    # Discovery defaults
    DEFAULT_TOPICS: list[str] = [
        "diffusion models",
        "flow matching",
        "LLM reasoning",
        "GRPO RLHF",
        "model quantization",
    ]
    DEFAULT_ARXIV_CATEGORIES: list[str] = ["cs.AI", "cs.LG", "cs.CV"]
    DEFAULT_TOP_N: int = 10

    # Ranking weights
    WEIGHT_CITATIONS: float = 0.3
    WEIGHT_RELEVANCE: float = 0.3
    WEIGHT_FRESHNESS: float = 0.2
    WEIGHT_NO_CODE: float = 0.2  # bonus for papers without code impl


config = AppConfig()
api_config = APIConfig()
