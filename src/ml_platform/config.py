"""ML Research Platform — Configuration management.

Provides centralized configuration for API keys, rate limits,
directory paths, and default settings used across the platform.

All settings can be overridden via environment variables or a .env file
in the project root. See .env.example for available options.

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

    All defaults can be overridden via environment variables.
    """

    # Semantic Scholar
    SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    SEMANTIC_SCHOLAR_BASE_URL: str = "https://api.semanticscholar.org/graph/v1"
    SEMANTIC_SCHOLAR_RATE_LIMIT: float = 1.0

    # arXiv
    ARXIV_BASE_URL: str = "https://export.arxiv.org/api/query"
    ARXIV_RATE_LIMIT: float = 1.0

    # HuggingFace Papers
    HUGGINGFACE_PAPERS_BASE_URL: str = "https://huggingface.co"
    HUGGINGFACE_PAPERS_RATE_LIMIT: float = 2.0

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
    GROBID_RATE_LIMIT: float = 1.0
    GROBID_TIMEOUT: float = 120.0

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Anthropic
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Google / Gemini
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))

    # Ollama (local inference, no API key needed)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen3:8b")


# ─── Platform-wide defaults (env-configurable) ──────────────────────

# LLM provider & model used across codegen, pipeline, CLI
DEFAULT_LLM_PROVIDER: str = os.getenv("ML_DEFAULT_LLM_PROVIDER", "ollama")
DEFAULT_LLM_MODEL: str = os.getenv(
    "ML_DEFAULT_LLM_MODEL",
    os.getenv("OLLAMA_DEFAULT_MODEL", "qwen3:8b"),
)

# Notion reporting
NOTION_API_KEY: str = os.getenv("NOTION_API_KEY", "")
NOTION_PARENT_PAGE_ID: str = os.getenv("NOTION_PARENT_PAGE_ID", "")

# GitHub push
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")


class AppConfig:
    """Application-level settings for directories, defaults, and weights."""

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
    WEIGHT_NO_CODE: float = 0.2


config = AppConfig()
api_config = APIConfig()
