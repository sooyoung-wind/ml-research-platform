"""Tests for Phase 0 — Project setup verification."""

from ml_platform import __version__
from ml_platform.config import config, api_config
from ml_platform.models import Author, Paper, PaperSource, ProcessingStatus
from ml_platform.db import PapersDB


def test_version():
    assert __version__ == "0.1.0"


def test_config_loads():
    assert config.PROJECT_ROOT.exists()
    assert config.DB_PATH.name == "papers.db"
    assert len(config.DEFAULT_TOPICS) > 0
    assert len(config.DEFAULT_ARXIV_CATEGORIES) > 0


def test_api_config_urls():
    assert "semanticscholar.org" in api_config.SEMANTIC_SCHOLAR_BASE_URL
    assert "arxiv.org" in api_config.ARXIV_BASE_URL
    assert "paperswithcode.com" in api_config.PWC_BASE_URL
    assert "openalex.org" in api_config.OPENALEX_BASE_URL


def test_paper_model():
    paper = Paper(
        paper_id="2301.07041",
        source=PaperSource.ARXIV,
        title="Test Paper",
        abstract="A test abstract.",
        arxiv_id="2301.07041",
        citation_count=42,
    )
    assert paper.has_code is False
    assert paper.citation_count == 42
    assert paper.status == ProcessingStatus.DISCOVERED


def test_paper_with_code():
    paper = Paper(
        paper_id="test-id",
        source=PaperSource.PAPERSWITHCODE,
        title="Paper with Code",
        code_url="https://github.com/test/repo",
    )
    assert paper.has_code is True


def test_author_model():
    author = Author(name="Test Author", affiliation="Test University")
    assert author.name == "Test Author"
    assert author.affiliation == "Test University"


def test_db_init(tmp_path):
    db = PapersDB(tmp_path / "test.db")
    stats = db.get_stats()
    assert stats["total_papers"] == 0


def test_db_upsert_and_get(tmp_path):
    db = PapersDB(tmp_path / "test.db")
    paper = Paper(
        paper_id="test-paper-1",
        source=PaperSource.ARXIV,
        title="Upsert Test",
        arxiv_id="2301.00001",
    )
    db.upsert_paper(paper)

    retrieved = db.get_paper("test-paper-1", PaperSource.ARXIV)
    assert retrieved is not None
    assert retrieved.title == "Upsert Test"
    assert retrieved.arxiv_id == "2301.00001"


def test_db_dedup(tmp_path):
    db = PapersDB(tmp_path / "test.db")
    paper = Paper(
        paper_id="dup-test",
        source=PaperSource.SEMANTIC_SCHOLAR,
        title="Original Title",
    )
    db.upsert_paper(paper)

    # Update with same ID
    paper_updated = Paper(
        paper_id="dup-test",
        source=PaperSource.SEMANTIC_SCHOLAR,
        title="Updated Title",
    )
    db.upsert_paper(paper_updated)

    stats = db.get_stats()
    assert stats["total_papers"] == 1

    retrieved = db.get_paper("dup-test", PaperSource.SEMANTIC_SCHOLAR)
    assert retrieved.title == "Updated Title"


def test_db_get_by_arxiv(tmp_path):
    db = PapersDB(tmp_path / "test.db")
    paper = Paper(
        paper_id="s2-123",
        source=PaperSource.SEMANTIC_SCHOLAR,
        title="Cross-ref Test",
        arxiv_id="2301.99999",
    )
    db.upsert_paper(paper)

    retrieved = db.get_paper_by_arxiv("2301.99999")
    assert retrieved is not None
    assert retrieved.title == "Cross-ref Test"
