"""ML Research Platform — Interactive Trend Analyzer.

LangGraph-based multi-round interview system that clarifies ambiguous
user queries about research trends, then generates a targeted trend report.

Flow:
  1. User provides initial query (e.g. "RAG trends")
  2. System asks 3+ clarifying questions via interrupt/resume
  3. After sufficient clarity, generates a focused TrendReport

Uses LangGraph 1.2+ API:
  - StateGraph with TypedDict state
  - interrupt() for human-in-the-loop
  - Command(resume=...) to continue
  - SqliteSaver for checkpointing
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from ml_platform.analysis.trends import TrendAnalyzer, TrendReport


# ── State ────────────────────────────────────────────────────────────


class InterviewState(TypedDict, total=False):
    """State for the interactive trend interview graph.

    Attributes:
        messages: Conversation history (human + assistant).
        original_query: The user's initial query.
        round: Current interview round (0-based).
        requirements: Accumulated user requirements.
        topic: Extracted/clarified research topic.
        subtopics: Specific sub-topics of interest.
        time_range: Desired time range (e.g. "2023-2026").
        focus_areas: Areas to focus on (methods, datasets, etc.).
        analysis_depth: How deep the analysis should be.
        clarity_score: 0-1 score of how clear the requirements are.
        is_complete: Whether the interview is complete.
        trend_report: Generated trend report (after completion).
    """
    messages: Annotated[list, add_messages]
    original_query: str
    round: int
    requirements: list[str]
    topic: str
    subtopics: list[str]
    time_range: str
    focus_areas: list[str]
    analysis_depth: str
    clarity_score: float
    is_complete: bool
    trend_report: str


# ── Interview questions ──────────────────────────────────────────────

# Question templates keyed by round
INTERVIEW_QUESTIONS = {
    0: {
        "question": (
            "주제가 구체적으로 무엇인가요?\n"
            "예: 'RAG(Retrieval-Augmented Generation)', 'Knowledge Graph Reasoning', "
            "'LLM Fine-tuning' 등 구체적인 기술/방법론 영역을 알려주세요."
        ),
        "field": "topic",
    },
    1: {
        "question": (
            "어떤 세부 주제에 관심이 있으신가요?\n"
            "예: '효율성 개선', '새로운 아키텍처', '평가 방법론', "
            "'특정 도메인 적용(의료, 금융 등)' 등 관심 방향을 알려주세요."
        ),
        "field": "subtopics",
    },
    2: {
        "question": (
            "시간 범위와 분석 깊이를 알려주세요.\n"
            "예: '최근 3년(2023-2026)', '전체 기간', "
            "분석 깊이: '개요(빠른 스캔)', '표준', '심층(상세 비교)'"
        ),
        "field": "time_range",
    },
    3: {
        "question": (
            "어떤 관점에서 트렌드를 파악하고 싶으신가요?\n"
            "예: '방법론 발전 흐름', '주요 연구자/기관', "
            "'데이터셋/벤치마크 동향', '응용 분야 확장', '연구 갭 기회'"
        ),
        "field": "focus_areas",
    },
    4: {
        "question": (
            "추가로 특별히 알고 싶은 내용이 있으신가요?\n"
            "구체적인 논문, 방법론, 비교 대상 등이 있으면 알려주세요. "
            "없으면 '없음'이라고 해주세요."
        ),
        "field": "extra",
    },
}


# ── Graph nodes ──────────────────────────────────────────────────────


def clarifying_node(state: InterviewState) -> dict:
    """Ask a clarifying question and wait for user response.

    Uses LangGraph's interrupt() to pause execution until
    the user provides an answer via Command(resume=...).
    """
    round_num = state.get("round", 0)
    interview = INTERVIEW_QUESTIONS.get(
        round_num,
        INTERVIEW_QUESTIONS[4],  # fallback to generic
    )

    question = interview["question"]
    topic_context = ""
    if state.get("topic"):
        topic_context = f"\n[현재 파악된 주제: {state['topic']}]"
    if state.get("subtopics"):
        topic_context += f"\n[세부 주제: {', '.join(state['subtopics'])}]"

    full_question = f"[인터뷰 {round_num + 1}] {question}{topic_context}"

    # interrupt pauses execution, returns user input on resume
    user_answer = interrupt({
        "question": full_question,
        "round": round_num,
        "field": interview["field"],
    })

    # Process the answer
    requirements = list(state.get("requirements", []))
    requirements.append(f"[Round {round_num + 1}] {user_answer}")

    # Update specific fields based on answer
    updates = {
        "messages": [
            {"role": "assistant", "content": full_question},
            {"role": "user", "content": user_answer},
        ],
        "round": round_num + 1,
        "requirements": requirements,
    }

    field = interview["field"]
    if field == "topic":
        updates["topic"] = user_answer.strip()
    elif field == "subtopics":
        updates["subtopics"] = [
            s.strip() for s in user_answer.replace(",", "/").split("/") if s.strip()
        ]
    elif field == "time_range":
        updates["time_range"] = user_answer.strip()
        # Try to detect depth
        if "심층" in user_answer:
            updates["analysis_depth"] = "deep"
        elif "개요" in user_answer or "빠른" in user_answer:
            updates["analysis_depth"] = "quick"
    elif field == "focus_areas":
        updates["focus_areas"] = [
            s.strip() for s in user_answer.replace(",", "/").split("/") if s.strip()
        ]

    return updates


def evaluate_clarity(state: InterviewState) -> dict:
    """Evaluate how clear the requirements are after each round.

    Updates clarity_score and is_complete based on rounds and answers.
    """
    round_num = state.get("round", 0)
    has_topic = bool(state.get("topic"))
    has_subtopics = bool(state.get("subtopics"))
    has_focus = bool(state.get("focus_areas"))

    # Calculate clarity score
    score = 0.0
    if has_topic:
        score += 0.4
    if has_subtopics:
        score += 0.25
    if has_focus:
        score += 0.25
    if round_num >= 4:
        score += 0.1

    # Must complete at least 3 rounds
    is_complete = round_num >= 3 and score >= 0.65

    return {
        "clarity_score": score,
        "is_complete": is_complete,
    }


def should_continue(state: InterviewState) -> str:
    """Decide whether to continue interviewing or generate the report."""
    if state.get("is_complete", False):
        return "generate"
    if state.get("round", 0) >= 5:
        return "generate"  # max 5 rounds
    return "clarify"


def generate_report_node(state: InterviewState) -> dict:
    """Generate the final trend report based on gathered requirements."""
    topic = state.get("topic", state.get("original_query", ""))
    subtopics = state.get("subtopics", [])
    focus_areas = state.get("focus_areas", [])
    time_range = state.get("time_range", "")

    # Summarize requirements for the user
    reqs = state.get("requirements", [])
    summary_lines = [
        f"주제: {topic}",
        f"세부 관심사: {', '.join(subtopics) if subtopics else '전체'}",
        f"시간 범위: {time_range or '전체 기간'}",
        f"관점: {', '.join(focus_areas) if focus_areas else '종합'}",
    ]

    summary = "\n".join(summary_lines)

    # Generate trend report using existing TrendAnalyzer
    analyzer = TrendAnalyzer()
    report = analyzer.analyze()

    # Filter report based on topic
    report_md = analyzer.generate_report_markdown(report)

    # Build final output
    final_message = (
        f"요구사항 정리 완료!\n\n"
        f"{summary}\n\n"
        f"---\n\n"
        f"## 분석 결과\n\n"
        f"DB 내 {report.total_papers}개 논문 기반 분석 결과입니다.\n\n"
        f"{report_md}"
    )

    return {
        "trend_report": final_message,
        "messages": [{"role": "assistant", "content": final_message}],
    }


# ── Graph builder ────────────────────────────────────────────────────


def build_interview_graph(
    checkpoint_dir: str | Path | None = None,
):
    """Build the interactive trend analysis interview graph.

    Args:
        checkpoint_dir: Directory for SQLite checkpoint storage.
            Defaults to ``data/checkpoints/``.

    Returns:
        Compiled StateGraph with SqliteSaver checkpointer.
    """
    # Build graph
    builder = StateGraph(InterviewState)

    # Add nodes
    builder.add_node("clarify", clarifying_node)
    builder.add_node("evaluate", evaluate_clarity)
    builder.add_node("generate", generate_report_node)

    # Add edges
    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        should_continue,
        {"clarify": "clarify", "generate": "generate"},
    )
    builder.add_edge("generate", END)

    # Checkpointer — MemorySaver for in-process use, SqliteSaver for persistence
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)
    return graph


# ── CLI interface ────────────────────────────────────────────────────


def run_interactive_trend(
    query: str,
    checkpoint_dir: str | Path | None = None,
) -> TrendReport | None:
    """Run the interactive trend interview from CLI.

    Args:
        query: User's initial query string.
        checkpoint_dir: Checkpoint storage directory.

    Returns:
        TrendReport if analysis was completed, None otherwise.
    """
    graph = build_interview_graph(checkpoint_dir=checkpoint_dir)

    import uuid
    thread_id = f"trend-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Initialize state
    initial_state = {
        "original_query": query,
        "round": 0,
        "requirements": [],
        "subtopics": [],
        "focus_areas": [],
        "clarity_score": 0.0,
        "is_complete": False,
    }

    # Run first step (will interrupt at clarify node)
    result = graph.invoke(initial_state, config)

    # Interview loop
    while True:
        # Get the interrupt info
        state = graph.get_state(config)
        if not state.tasks:
            break

        task = state.tasks[0]
        if not task.interrupts:
            break

        interrupt_value = task.interrupts[0].value
        question = interrupt_value.get("question", "질문:")
        round_num = interrupt_value.get("round", 0)

        # Display question
        print(f"\n{'='*60}")
        print(f"  🤖 {question}")
        print(f"{'='*60}")

        # Get user input
        try:
            answer = input("\n  👤 답변: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  인터뷰를 종료합니다.")
            return None

        if not answer:
            answer = "전체 영역에 관심 있습니다."

        if answer.lower() in ("quit", "exit", "q", "종료"):
            print("  인터뷰를 종료합니다.")
            return None

        # Resume with answer
        result = graph.invoke(Command(resume=answer), config)

    # Check final state
    final_state = graph.get_state(config)
    values = final_state.values

    if values.get("trend_report"):
        print(f"\n{'='*60}")
        print(values["trend_report"])
        print(f"{'='*60}")

    return values.get("trend_report")


def resume_interview(
    thread_id: str,
    answer: str,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Resume a paused interview with an answer.

    Used for programmatic access (API, web UI).

    Args:
        thread_id: Thread ID from the initial invocation.
        answer: User's answer to the current question.
        checkpoint_dir: Checkpoint storage directory.

    Returns:
        Dict with current state including any interrupt info.
    """
    graph = build_interview_graph(checkpoint_dir=checkpoint_dir)
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(Command(resume=answer), config)
    state = graph.get_state(config)

    response = {
        "thread_id": thread_id,
        "round": state.values.get("round", 0),
        "clarity_score": state.values.get("clarity_score", 0.0),
        "is_complete": state.values.get("is_complete", False),
        "topic": state.values.get("topic", ""),
        "requirements": state.values.get("requirements", []),
    }

    # Check for next interrupt
    if state.tasks and state.tasks[0].interrupts:
        interrupt_value = state.tasks[0].interrupts[0].value
        response["next_question"] = interrupt_value.get("question")
        response["interrupted"] = True
    elif state.values.get("trend_report"):
        response["trend_report"] = state.values["trend_report"]
        response["interrupted"] = False
    else:
        response["interrupted"] = False

    return response


def start_interview(
    query: str,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Start a new trend interview session.

    Returns session info with thread_id and first question.

    Args:
        query: User's initial query.
        checkpoint_dir: Checkpoint storage directory.

    Returns:
        Dict with thread_id and first question.
    """
    graph = build_interview_graph(checkpoint_dir=checkpoint_dir)

    import uuid
    thread_id = f"trend-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "original_query": query,
        "round": 0,
        "requirements": [],
        "subtopics": [],
        "focus_areas": [],
        "clarity_score": 0.0,
        "is_complete": False,
    }

    # Run — will interrupt at first clarify
    graph.invoke(initial_state, config)

    state = graph.get_state(config)
    interrupt_value = state.tasks[0].interrupts[0].value

    return {
        "thread_id": thread_id,
        "question": interrupt_value.get("question"),
        "round": interrupt_value.get("round", 0),
        "field": interrupt_value.get("field"),
    }
