"""ML Research Platform — Agent package.

Interactive research agent that guides users from question to code.
"""

from ml_platform.agent.question_clarifier import QuestionClarifier, ClarificationResult
from ml_platform.agent.research_interviewer import ResearchInterviewer, SearchStrategy
from ml_platform.agent.session import ResearchSession, SessionState, SessionPhase
from ml_platform.agent.unified_search import UnifiedSearcher
from ml_platform.agent.dashboard import DashboardGenerator
from ml_platform.agent.reproducibility import ReproducibilityAnalyzer, ReproducibilityReport
from ml_platform.agent.future_work import FutureWorkPlanner, FutureWorkReport

__all__ = [
    "QuestionClarifier",
    "ClarificationResult",
    "ResearchInterviewer",
    "SearchStrategy",
    "ResearchSession",
    "SessionState",
    "SessionPhase",
    "UnifiedSearcher",
    "DashboardGenerator",
    "ReproducibilityAnalyzer",
    "ReproducibilityReport",
    "FutureWorkPlanner",
    "FutureWorkReport",
]
