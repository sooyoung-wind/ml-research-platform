"""ML Research Platform — SQLite database layer.

Provides PapersDB, a SQLite-backed store for papers and pipeline state
with upsert, query, and statistics capabilities.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from ml_platform.config import config
from ml_platform.models import Author, Paper, PaperSource, ProcessingStatus


class PapersDB:
    """SQLite-backed storage for papers and pipeline state.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path | None = None):
        """Initialize the database and create tables if needed.

        Args:
            db_path: Path to the SQLite database file. Defaults to
                ``config.DB_PATH``.
        """
        self.db_path = db_path or config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a database connection with WAL mode and foreign keys enabled.

        Yields:
            A sqlite3.Connection with row factory and pragmas configured.

        Raises:
            sqlite3.Error: If the database connection cannot be established.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create database tables and indexes if they do not exist."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    doi TEXT,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    authors_json TEXT,
                    published_date TEXT,
                    venue TEXT,
                    year INTEGER,
                    citation_count INTEGER,
                    relevance_score REAL,
                    arxiv_id TEXT,
                    url TEXT,
                    pdf_url TEXT,
                    code_url TEXT,
                    pwc_id TEXT,
                    categories_json TEXT,
                    keywords_json TEXT,
                    status TEXT DEFAULT 'discovered',
                    local_pdf_path TEXT,
                    parsed_content_json TEXT,
                    composite_score REAL,
                    discovered_at TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (paper_id, source)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_papers_status
                ON papers(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id
                ON papers(arxiv_id)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discovery_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    total_found INTEGER,
                    paper_ids_json TEXT,
                    duration_seconds REAL,
                    timestamp TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    paper_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    analysis_json TEXT NOT NULL,
                    summary TEXT,
                    key_contributions_json TEXT,
                    methodology_type TEXT,
                    domain TEXT,
                    model_used TEXT,
                    status TEXT DEFAULT 'pending',
                    analyzed_at TEXT,
                    self_correction_applied INTEGER DEFAULT 0,
                    correction_notes TEXT,
                    PRIMARY KEY (paper_id, source)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_registry (
                    canonical_id TEXT PRIMARY KEY,
                    doi TEXT,
                    arxiv_id TEXT,
                    title TEXT,
                    title_variants_json TEXT,
                    source_ids_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_doi ON entity_registry(doi)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_arxiv ON entity_registry(arxiv_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_status ON analysis_results(status)
            """)

    def upsert_paper(self, paper: Paper) -> None:
        """Insert or update a paper.

        Args:
            paper: The Paper object to upsert.
        """
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO papers (
                    paper_id, source, doi, title, abstract, authors_json,
                    published_date, venue, year, citation_count, relevance_score,
                    arxiv_id, url, pdf_url, code_url, pwc_id,
                    categories_json, keywords_json, status, local_pdf_path,
                    parsed_content_json, composite_score, discovered_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, source) DO UPDATE SET
                    doi=excluded.doi,
                    title=excluded.title,
                    abstract=excluded.abstract,
                    authors_json=excluded.authors_json,
                    citation_count=excluded.citation_count,
                    relevance_score=excluded.relevance_score,
                    code_url=excluded.code_url,
                    composite_score=excluded.composite_score,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    paper.paper_id,
                    paper.source.value,
                    paper.doi,
                    paper.title,
                    paper.abstract,
                    json.dumps([a.model_dump() for a in paper.authors]),
                    paper.published_date.isoformat() if paper.published_date else None,
                    paper.venue,
                    paper.year,
                    paper.citation_count,
                    paper.relevance_score,
                    paper.arxiv_id,
                    paper.url,
                    paper.pdf_url,
                    paper.code_url,
                    paper.pwc_id,
                    json.dumps(paper.categories),
                    json.dumps(paper.keywords),
                    paper.status.value,
                    paper.local_pdf_path,
                    json.dumps(paper.parsed_content) if paper.parsed_content else None,
                    paper.composite_score,
                    paper.discovered_at.isoformat(),
                    datetime.now().isoformat(),
                ),
            )

    def upsert_papers(self, papers: list[Paper]) -> int:
        """Bulk upsert papers.

        Args:
            papers: List of Paper objects to upsert.

        Returns:
            Count of upserted papers.
        """
        for paper in papers:
            self.upsert_paper(paper)
        return len(papers)

    def get_paper(self, paper_id: str, source: PaperSource) -> Paper | None:
        """Get a paper by ID and source.

        Args:
            paper_id: The paper's primary ID.
            source: The paper's source.

        Returns:
            The Paper object, or None if not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE paper_id = ? AND source = ?",
                (paper_id, source.value),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_paper(row)

    def get_paper_by_arxiv(self, arxiv_id: str) -> Paper | None:
        """Get a paper by arXiv ID.

        Args:
            arxiv_id: The arXiv identifier.

        Returns:
            The Paper object, or None if not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_paper(row)

    def get_papers(
        self,
        status: ProcessingStatus | None = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "composite_score DESC",
    ) -> list[Paper]:
        """Query papers with optional filters.

        Args:
            status: Filter by processing status.
            limit: Maximum number of papers to return.
            offset: Number of papers to skip.
            order_by: SQL ORDER BY clause.

        Returns:
            List of matching Paper objects.
        """
        query = "SELECT * FROM papers"
        params: list = []

        if status:
            query += " WHERE status = ?"
            params.append(status.value)

        query += f" ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_paper(r) for r in rows]

    def update_status(self, paper_id: str, source: PaperSource, status: ProcessingStatus) -> None:
        """Update paper processing status.

        Args:
            paper_id: The paper's primary ID.
            source: The paper's source.
            status: The new processing status.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE papers SET status = ?, updated_at = ? WHERE paper_id = ? AND source = ?",
                (status.value, datetime.now().isoformat(), paper_id, source.value),
            )

    def paper_exists(self, paper_id: str, source: PaperSource) -> bool:
        """Check if a paper already exists in the DB.

        Args:
            paper_id: The paper's primary ID.
            source: The paper's source.

        Returns:
            True if the paper exists.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE paper_id = ? AND source = ?",
                (paper_id, source.value),
            ).fetchone()
            return row is not None

    def log_discovery(self, query: str, total_found: int, paper_ids: list[str], duration: float) -> None:
        """Log a discovery run.

        Args:
            query: The search query used.
            total_found: Total number of papers found.
            paper_ids: List of paper IDs discovered.
            duration: Duration of the discovery in seconds.
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO discovery_log (query, total_found, paper_ids_json, duration_seconds, timestamp) VALUES (?, ?, ?, ?, ?)",
                (query, total_found, json.dumps(paper_ids), duration, datetime.now().isoformat()),
            )

    def save_analysis(self, paper_id: str, source: str, analysis: 'PaperAnalysis') -> None:
        """Save paper analysis results."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO analysis_results (
                    paper_id, source, analysis_json, summary, key_contributions_json,
                    methodology_type, domain, model_used, status, analyzed_at,
                    self_correction_applied, correction_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id, source) DO UPDATE SET
                    analysis_json=excluded.analysis_json,
                    summary=excluded.summary,
                    key_contributions_json=excluded.key_contributions_json,
                    methodology_type=excluded.methodology_type,
                    domain=excluded.domain,
                    model_used=excluded.model_used,
                    status=excluded.status,
                    analyzed_at=excluded.analyzed_at,
                    self_correction_applied=excluded.self_correction_applied,
                    correction_notes=excluded.correction_notes
                """,
                (
                    paper_id,
                    source,
                    analysis.model_dump_json(),
                    analysis.summary,
                    json.dumps(analysis.key_contributions),
                    analysis.methodology_type,
                    analysis.domain,
                    analysis.model_used,
                    analysis.status.value,
                    analysis.analyzed_at.isoformat(),
                    1 if analysis.self_correction_applied else 0,
                    analysis.correction_notes,
                ),
            )

    def get_analysis(self, paper_id: str, source: str) -> dict | None:
        """Get analysis results for a paper."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_results WHERE paper_id = ? AND source = ?",
                (paper_id, source),
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def get_analysis_object(self, paper_id: str, source: str) -> 'PaperAnalysis | None':
        """Get analysis results as a PaperAnalysis object."""
        data = self.get_analysis(paper_id, source)
        if data is None:
            return None
        from ml_platform.analysis.models import PaperAnalysis
        return PaperAnalysis.model_validate_json(data['analysis_json'])

    def has_analysis(self, paper_id: str, source: str) -> bool:
        """Check if analysis exists for a paper."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM analysis_results WHERE paper_id = ? AND source = ?",
                (paper_id, source),
            ).fetchone()
            return row is not None

    def register_entity(self, canonical_id: str, doi: str | None, arxiv_id: str | None, title: str, source_id: str) -> None:
        """Register or update a canonical entity in the registry."""
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT source_ids_json FROM entity_registry WHERE canonical_id = ?",
                (canonical_id,),
            ).fetchone()

            if existing:
                source_ids = json.loads(existing['source_ids_json']) if existing['source_ids_json'] else []
                if source_id not in source_ids:
                    source_ids.append(source_id)
                conn.execute(
                    """UPDATE entity_registry SET
                       source_ids_json=?, updated_at=?
                       WHERE canonical_id=?""",
                    (json.dumps(source_ids), datetime.now().isoformat(), canonical_id),
                )
            else:
                conn.execute(
                    """INSERT INTO entity_registry
                       (canonical_id, doi, arxiv_id, title, source_ids_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        canonical_id,
                        doi,
                        arxiv_id,
                        title,
                        json.dumps([source_id]),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ),
                )

    def resolve_entity(self, doi: str | None = None, arxiv_id: str | None = None) -> str | None:
        """Look up a canonical ID by DOI or arXiv ID."""
        with self._conn() as conn:
            if doi:
                row = conn.execute(
                    "SELECT canonical_id FROM entity_registry WHERE doi = ?",
                    (doi,),
                ).fetchone()
                if row:
                    return row['canonical_id']
            if arxiv_id:
                row = conn.execute(
                    "SELECT canonical_id FROM entity_registry WHERE arxiv_id = ?",
                    (arxiv_id,),
                ).fetchone()
                if row:
                    return row['canonical_id']
        return None

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dict with keys: total_papers, with_code, without_code, by_status.
        """
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            by_status = dict(
                conn.execute("SELECT status, COUNT(*) FROM papers GROUP BY status").fetchall()
            )
            with_code = conn.execute("SELECT COUNT(*) FROM papers WHERE code_url IS NOT NULL").fetchone()[0]
            return {
                "total_papers": total,
                "with_code": with_code,
                "without_code": total - with_code,
                "by_status": by_status,
            }

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Paper:
        """Convert a database row to a Paper model.

        Args:
            row: A sqlite3.Row from the papers table.

        Returns:
            A Paper instance populated from the row data.
        """
        authors_data = json.loads(row["authors_json"]) if row["authors_json"] else []
        return Paper(
            paper_id=row["paper_id"],
            source=PaperSource(row["source"]),
            doi=row["doi"],
            title=row["title"],
            abstract=row["abstract"],
            authors=[Author(**a) for a in authors_data],
            published_date=datetime.fromisoformat(row["published_date"]) if row["published_date"] else None,
            venue=row["venue"],
            year=row["year"],
            citation_count=row["citation_count"],
            relevance_score=row["relevance_score"],
            arxiv_id=row["arxiv_id"],
            url=row["url"],
            pdf_url=row["pdf_url"],
            code_url=row["code_url"],
            pwc_id=row["pwc_id"],
            categories=json.loads(row["categories_json"]) if row["categories_json"] else [],
            keywords=json.loads(row["keywords_json"]) if row["keywords_json"] else [],
            status=ProcessingStatus(row["status"]),
            local_pdf_path=row["local_pdf_path"],
            parsed_content=json.loads(row["parsed_content_json"]) if row["parsed_content_json"] else None,
            composite_score=row["composite_score"],
            discovered_at=datetime.fromisoformat(row["discovered_at"]) if row["discovered_at"] else datetime.now(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.now(),
        )
