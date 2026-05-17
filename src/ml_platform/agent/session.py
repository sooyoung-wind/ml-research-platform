"""ML Research Platform — Research Session Manager.

Manages the full lifecycle of a research session:
  Question → Interview → Search → Build → Report → Select → Implement → Analyze

State machine with 9 phases, each building on the previous.
Supports save/resume via JSON checkpointing.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ml_platform.config import config


class SessionPhase(str, Enum):
    """Phases of a research session."""
    QUESTION = "question"           # Initial question received
    INTERVIEW = "interview"         # Clarifying interview in progress
    SEARCHING = "searching"         # Multi-source paper collection
    BUILDING = "building"           # KG + Wiki construction
    REPORTING = "reporting"         # Dashboard generation
    SELECTING = "selecting"         # User paper selection
    IMPLEMENTING = "implementing"   # DeepCode execution
    ANALYZING = "analyzing"         # Reproducibility analysis
    COMPLETED = "completed"         # Session finished


@dataclass
class SessionState:
    """Complete state of a research session.

    Attributes:
        session_id: Unique session identifier.
        created_at: Session creation timestamp.
        updated_at: Last update timestamp.
        phase: Current session phase.
        original_question: User's original question.
        interview_rounds: Interview Q&A pairs.
        search_strategy: Derived search strategy.
        discovered_papers: Papers found during search.
        selected_papers: Papers selected by user.
        kg_stats: Knowledge graph statistics.
        wiki_stats: Wiki statistics.
        dashboard_path: Path to generated dashboard HTML.
        codegen_results: DeepCode execution results.
        reproducibility_report: Reproducibility analysis.
        future_work: Future work suggestions.
        error: Last error message (if any).
    """
    session_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    phase: SessionPhase = SessionPhase.QUESTION
    original_question: str = ""
    interview_rounds: list[dict] = field(default_factory=list)
    search_strategy: dict = field(default_factory=dict)
    discovered_papers: list[dict] = field(default_factory=list)
    selected_papers: list[str] = field(default_factory=list)
    kg_stats: dict = field(default_factory=dict)
    wiki_stats: dict = field(default_factory=dict)
    dashboard_path: str = ""
    codegen_results: list[dict] = field(default_factory=list)
    reproducibility_report: dict = field(default_factory=dict)
    future_work: list[dict] = field(default_factory=list)
    error: str = ""


class ResearchSession:
    """Manages a complete research session lifecycle.

    Usage:
        session = ResearchSession()
        session.start("RAG hallucination 연구")
        session.run_interview(answers=["특정 답변", ...])
        session.run_search()
        session.build_knowledge()
        session.generate_dashboard()
        session.select_papers(["2605.13153", "2312.00752"])
        session.implement_selected()
        session.analyze_reproducibility()
        session.plan_future_work()
    """

    def __init__(self, session_dir: Path | None = None) -> None:
        self.session_dir = session_dir or config.DATA_DIR / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.state = SessionState()

    def start(self, question: str) -> SessionState:
        """Start a new research session with the given question."""
        self.state = SessionState(
            session_id=uuid.uuid4().hex[:12],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            phase=SessionPhase.QUESTION,
            original_question=question,
        )
        self._save()
        return self.state

    def load(self, session_id: str) -> SessionState:
        """Load an existing session by ID."""
        path = self.session_dir / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.state = SessionState(
            session_id=data["session_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            phase=SessionPhase(data["phase"]),
            original_question=data["original_question"],
            interview_rounds=data.get("interview_rounds", []),
            search_strategy=data.get("search_strategy", {}),
            discovered_papers=data.get("discovered_papers", []),
            selected_papers=data.get("selected_papers", []),
            kg_stats=data.get("kg_stats", {}),
            wiki_stats=data.get("wiki_stats", {}),
            dashboard_path=data.get("dashboard_path", ""),
            codegen_results=data.get("codegen_results", []),
            reproducibility_report=data.get("reproducibility_report", {}),
            future_work=data.get("future_work", []),
            error=data.get("error", ""),
        )
        return self.state

    def _save(self) -> None:
        """Save session state to disk."""
        self.state.updated_at = datetime.now().isoformat()
        path = self.session_dir / f"{self.state.session_id}.json"
        path.write_text(json.dumps({
            "session_id": self.state.session_id,
            "created_at": self.state.created_at,
            "updated_at": self.state.updated_at,
            "phase": self.state.phase.value,
            "original_question": self.state.original_question,
            "interview_rounds": self.state.interview_rounds,
            "search_strategy": self.state.search_strategy,
            "discovered_papers": self.state.discovered_papers,
            "selected_papers": self.state.selected_papers,
            "kg_stats": self.state.kg_stats,
            "wiki_stats": self.state.wiki_stats,
            "dashboard_path": self.state.dashboard_path,
            "codegen_results": self.state.codegen_results,
            "reproducibility_report": self.state.reproducibility_report,
            "future_work": self.state.future_work,
            "error": self.state.error,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_sessions(self) -> list[dict]:
        """List all saved sessions."""
        sessions = []
        for path in sorted(self.session_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data["session_id"],
                    "created_at": data["created_at"],
                    "phase": data["phase"],
                    "question": data["original_question"][:80],
                    "papers": len(data.get("discovered_papers", [])),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def advance(self, phase: SessionPhase, **updates: Any) -> SessionState:
        """Advance to a new phase with optional state updates."""
        self.state.phase = phase
        for key, value in updates.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        self._save()
        return self.state
