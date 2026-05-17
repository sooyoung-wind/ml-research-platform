"""ML Research Platform — Interactive Trend Analyzer.

LangGraph-based multi-round interview system that clarifies ambiguous
user queries about research trends, then generates a targeted trend report.

Flow:
  1. User provides initial query (e.g. "DDPM 시계열 보간")
  2. LLM generates contextual clarifying questions via interrupt/resume
  3. After sufficient clarity (3+ rounds), searches papers & generates report

Uses LangGraph 1.2+ API:
  - StateGraph with TypedDict state
  - interrupt() for human-in-the-loop
  - Command(resume=...) to continue
  - MemorySaver for checkpointing
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from ml_platform.analysis.trends import TrendAnalyzer, TrendReport
from ml_platform.config import APIConfig


# ── State ────────────────────────────────────────────────────────────


class InterviewState(TypedDict, total=False):
    """State for the interactive trend interview graph."""
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
    search_keywords: list[str]


# ── LLM helpers ──────────────────────────────────────────────────────


def _get_llm_client():
    """Get an OpenAI-compatible client for the configured LLM."""
    from openai import OpenAI
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("ML_DEFAULT_LLM_MODEL", "gemma4:31b-cloud")
    client = OpenAI(base_url=base_url + "/v1", api_key="unused")
    return client, model


def _llm_generate_question(
    original_query: str,
    round_num: int,
    requirements: list[str],
    topic: str | None,
    subtopics: list[str],
    focus_areas: list[str],
    time_range: str | None,
) -> tuple[str, str]:
    """Ask the LLM to generate a context-aware clarifying question.

    Returns:
        (question_text, field_name) tuple.
    """
    client, model = _get_llm_client()

    # Build context for the LLM
    context_parts = [f"원본 쿼리: \"{original_query}\""]
    if topic:
        context_parts.append(f"파악된 주제: {topic}")
    if subtopics:
        context_parts.append(f"세부 관심사: {', '.join(subtopics)}")
    if focus_areas:
        context_parts.append(f"관심 관점: {', '.join(focus_areas)}")
    if time_range:
        context_parts.append(f"시간 범위: {time_range}")
    context_parts.append(f"현재 인터뷰 라운드: {round_num + 1}/5")
    context_parts.append(f"지금까지 수집된 요구사항: {json.dumps(requirements, ensure_ascii=False)}")

    context = "\n".join(context_parts)

    # Determine what to ask based on round
    field_guides = {
        0: ("topic", "사용자가 관심 있는 구체적인 기술/방법론/연구 분야를 파악하세요. 아직 정보가 거의 없으므로 넓게 질문하세요."),
        1: ("subtopics", "구체적인 세부 방향, 관심 있는 특정 문제나 접근법을 파악하세요."),
        2: ("time_range", "시간 범위(예: 최근 3년, 2020-2026 등)와 분석 깊이(개요/표준/심층)를 파악하세요."),
        3: ("focus_areas", "어떤 관점에서 분석하고 싶은지 파악하세요: 방법론 발전, 성능 비교, 연구 갭, 주요 연구자 등."),
        4: ("extra", "추가로 알고 싶은 내용이 있는지 물어보세요. 없으면 '없음'이라고 하도록 안내하세요."),
    }

    field, guide = field_guides.get(min(round_num, 4), ("extra", "추가 정보가 필요한지 물어보세요."))

    system_prompt = f"""당신은 ML/DL 연구 트렌드 분석을 돕는 전문 인터뷰어입니다.
사용자의 애매모호한 쿼리를 정확한 분석 요구사항으로 구체화하는 것이 목표입니다.

규칙:
- 한국어로 자연스럽게 대화하세요
- 한 번에 하나의 질문만 하세요
- 이전 대화 내용을 반영하여 맥락 있는 질문을 하세요
- 구체적인 예시를 제시하세요
- 2-3문장으로 간결하게
- 목적: {guide}

현재 컨텍스트:
{context}"""

    user_prompt = "사용자에게 할 질문을 생성하세요. 질문만 출력하세요 (설명 불필요)."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        question = response.choices[0].message.content.strip()
        # Clean up — remove quotes if wrapped
        question = question.strip('"').strip("'")
    except Exception as e:
        # Fallback to template questions
        fallback = {
            0: "관심 있는 구체적인 기술이나 연구 분야를 알려주세요. 예: DDPM, RAG, Knowledge Graph 등",
            1: "어떤 세부 방향에 관심이 있으신가요? 예: 성능 개선, 새로운 아키텍처, 특정 도메인 적용 등",
            2: "분석할 시간 범위와 깊이를 알려주세요. 예: 최근 3년, 심층 분석 등",
            3: "어떤 관점에서 트렌드를 보고 싶으신가요? 예: 방법론 발전, 성능 비교, 연구 갭 등",
            4: "추가로 특별히 알고 싶은 내용이 있으신가요?",
        }
        question = fallback.get(min(round_num, 4), "추가 정보를 알려주세요.")
        field = field_guides.get(min(round_num, 4), ("extra", ""))[0]
        print(f"  [WARN] LLM 질문 생성 실패, 폴백 사용: {e}")

    return question, field


def _llm_extract_keywords(
    original_query: str,
    requirements: list[str],
    topic: str,
    subtopics: list[str],
) -> list[str]:
    """Use LLM to extract search keywords from gathered requirements.

    Returns:
        List of 3-7 search keywords/phrases.
    """
    client, model = _get_llm_client()

    context = f"""원본 쿼리: {original_query}
파악된 주제: {topic}
세부 관심사: {', '.join(subtopics) if subtopics else '전체'}
수집된 요구사항: {json.dumps(requirements, ensure_ascii=False)}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """학술 논문 검색에 사용할 키워드를 추출하세요.
규칙:
- 영어 키워드 3-7개를 생성
- 가능하면 기술 용어의 full name과 약어 모두 포함 (예: "diffusion model" AND "DDPM")
- JSON 배열 형식으로만 출력
- 예: ["diffusion model", "time series", "imputation", "missing data", "DDPM"]""",
                },
                {"role": "user", "content": context},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()
        # Extract JSON array from response
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            keywords = json.loads(match.group())
            return [k.strip().strip('"') for k in keywords if isinstance(k, str)]
    except Exception as e:
        print(f"  [WARN] 키워드 추출 실패: {e}")

    # Fallback: use topic words
    words = re.findall(r'[A-Za-z]+(?:\s+[A-Za-z]+)*', topic)
    return words[:5] if words else [topic]


def _llm_generate_summary(requirements: list, report_md: str) -> str:
    """Use LLM to add a final summary tailored to user's interest."""
    client, model = _get_llm_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """사용자의 관심사에 맞게 트렌드 분석 결과를 요약하세요.
- 한국어로 작성
- 핵심 인사이트 3-5개를 불릿으로 정리
- 연구 방향 제안 포함
- 500자 이내""",
                },
                {
                    "role": "user",
                    "content": f"사용자 요구사항:\n{json.dumps(requirements, ensure_ascii=False)}\n\n분석 결과:\n{report_md[:3000]}",
                },
            ],
            max_tokens=800,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


# ── Graph nodes ──────────────────────────────────────────────────────


def clarifying_node(state: InterviewState) -> dict:
    """Ask a clarifying question generated by LLM and wait for user response."""
    round_num = state.get("round", 0)

    # Generate contextual question via LLM
    question, field = _llm_generate_question(
        original_query=state.get("original_query", ""),
        round_num=round_num,
        requirements=state.get("requirements", []),
        topic=state.get("topic", None),
        subtopics=state.get("subtopics", []),
        focus_areas=state.get("focus_areas", []),
        time_range=state.get("time_range", None),
    )

    # Build context display
    context_lines = []
    if state.get("topic"):
        context_lines.append(f"[파악된 주제: {state['topic']}]")
    if state.get("subtopics"):
        context_lines.append(f"[세부 관심사: {', '.join(state['subtopics'])}]")
    if state.get("time_range"):
        context_lines.append(f"[시간 범위: {state['time_range']}]")

    context_str = "\n".join(context_lines)
    full_question = f"[인터뷰 {round_num + 1}] {question}"
    if context_str:
        full_question += f"\n{context_str}"

    # interrupt pauses execution, returns user input on resume
    user_answer = interrupt({
        "question": full_question,
        "round": round_num,
        "field": field,
    })

    # Process the answer
    requirements = list(state.get("requirements", []))
    requirements.append(f"[Round {round_num + 1}] {user_answer}")

    updates = {
        "messages": [
            {"role": "assistant", "content": full_question},
            {"role": "user", "content": user_answer},
        ],
        "round": round_num + 1,
        "requirements": requirements,
    }

    if field == "topic":
        updates["topic"] = user_answer.strip()
    elif field == "subtopics":
        updates["subtopics"] = [
            s.strip() for s in re.split(r'[,/]', user_answer) if s.strip()
        ]
    elif field == "time_range":
        updates["time_range"] = user_answer.strip()
        if "심층" in user_answer:
            updates["analysis_depth"] = "deep"
        elif "개요" in user_answer or "빠른" in user_answer:
            updates["analysis_depth"] = "quick"
    elif field == "focus_areas":
        updates["focus_areas"] = [
            s.strip() for s in re.split(r'[,/]', user_answer) if s.strip()
        ]

    return updates


def evaluate_clarity(state: InterviewState) -> dict:
    """Evaluate how clear the requirements are after each round."""
    round_num = state.get("round", 0)
    has_topic = bool(state.get("topic"))
    has_subtopics = bool(state.get("subtopics"))
    has_focus = bool(state.get("focus_areas"))
    has_time = bool(state.get("time_range"))

    score = 0.0
    if has_topic:
        score += 0.35
    if has_subtopics:
        score += 0.20
    if has_time:
        score += 0.15
    if has_focus:
        score += 0.20
    if round_num >= 4:
        score += 0.10

    is_complete = round_num >= 3 and score >= 0.70

    return {
        "clarity_score": score,
        "is_complete": is_complete,
    }


def should_continue(state: InterviewState) -> str:
    """Decide whether to continue interviewing or generate the report."""
    if state.get("is_complete", False):
        return "generate"
    if state.get("round", 0) >= 5:
        return "generate"
    return "clarify"


def generate_report_node(state: InterviewState) -> dict:
    """Generate the final trend report based on gathered requirements.

    1. Extract search keywords from requirements via LLM
    2. Search for relevant papers using discovery module
    3. Run trend analysis on found papers
    4. Generate tailored report with LLM summary
    """
    topic = state.get("topic", state.get("original_query", ""))
    subtopics = state.get("subtopics", [])
    focus_areas = state.get("focus_areas", [])
    time_range = state.get("time_range", "")
    requirements = state.get("requirements", [])

    # Step 1: Extract search keywords
    print(f"  [INFO] 키워드 추출 중...")
    keywords = _llm_extract_keywords(
        original_query=state.get("original_query", ""),
        requirements=requirements,
        topic=topic,
        subtopics=subtopics,
    )
    print(f"  [INFO] 검색 키워드: {keywords}")

    # Step 2: Search for relevant papers
    relevant_papers = []
    try:
        from ml_platform.db import PapersDB
        from ml_platform.models import Paper
        db = PapersDB()

        # Search DB papers by keyword matching
        all_papers = db.get_papers(limit=1000)

        search_text = " ".join(keywords).lower()
        topic_lower = topic.lower()

        for paper in all_papers:
            paper_text = f"{paper.title} {' '.join(paper.categories)}".lower()
            # Score by keyword matches
            score = sum(1 for kw in keywords if kw.lower() in paper_text)
            # Boost if topic words match
            topic_words = re.findall(r'[a-z]+', topic_lower)
            score += sum(1 for w in topic_words if w in paper_text)
            if score > 0:
                relevant_papers.append((paper, score))

        relevant_papers.sort(key=lambda x: x[1], reverse=True)
        relevant_papers = [p for p, s in relevant_papers]

        print(f"  [INFO] DB에서 {len(relevant_papers)}개 관련 논문 발견")
    except Exception as e:
        print(f"  [WARN] 논문 검색 실패: {e}")

    # Step 3: Run trend analysis
    print(f"  [INFO] 트렌드 분석 중...")
    analyzer = TrendAnalyzer()

    if relevant_papers:
        report = analyzer.analyze(papers=relevant_papers)
    else:
        report = analyzer.analyze()

    report_md = analyzer.generate_report_markdown(report)

    # Step 4: Generate LLM summary
    print(f"  [INFO] 맞춤 요약 생성 중...")
    llm_summary = _llm_generate_summary(requirements, report_md)

    # Build final output
    summary_lines = [
        f"주제: {topic}",
        f"세부 관심사: {', '.join(subtopics) if subtopics else '전체'}",
        f"시간 범위: {time_range or '전체 기간'}",
        f"관점: {', '.join(focus_areas) if focus_areas else '종합'}",
        f"검색 키워드: {', '.join(keywords)}",
        f"분석 대상 논문: {len(relevant_papers)}개 (DB 내 관련 논문)",
    ]

    final_message = (
        f"요구사항 정리 완료!\n\n"
        + "\n".join(summary_lines)
        + f"\n\n---\n"
    )

    if llm_summary:
        final_message += f"\n## 맞춤 인사이트\n\n{llm_summary}\n\n---\n"

    final_message += (
        f"\n## 상세 트렌드 분석\n\n"
        f"총 {report.total_papers}개 논문 기반 | "
        f"{report.year_range[0]}–{report.year_range[1]}\n\n"
        f"{report_md}"
    )

    return {
        "trend_report": final_message,
        "search_keywords": keywords,
        "messages": [{"role": "assistant", "content": final_message}],
    }


# ── Graph builder ────────────────────────────────────────────────────


def build_interview_graph():
    """Build the interactive trend analysis interview graph."""
    builder = StateGraph(InterviewState)

    builder.add_node("clarify", clarifying_node)
    builder.add_node("evaluate", evaluate_clarity)
    builder.add_node("generate", generate_report_node)

    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        should_continue,
        {"clarify": "clarify", "generate": "generate"},
    )
    builder.add_edge("generate", END)

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# ── CLI interface ────────────────────────────────────────────────────


def run_interactive_trend(
    query: str,
    checkpoint_dir: str | Path | None = None,
) -> str | None:
    """Run the interactive trend interview from CLI.

    Args:
        query: User's initial query string.
        checkpoint_dir: Checkpoint storage directory (unused with MemorySaver).

    Returns:
        Trend report string if completed, None otherwise.
    """
    graph = build_interview_graph()

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

    # Run first step (will interrupt at clarify node)
    graph.invoke(initial_state, config)

    # Interview loop
    while True:
        state = graph.get_state(config)
        if not state.tasks:
            break

        task = state.tasks[0]
        if not task.interrupts:
            break

        interrupt_value = task.interrupts[0].value
        question = interrupt_value.get("question", "질문:")
        round_num = interrupt_value.get("round", 0)

        print(f"\n{'='*60}")
        print(f"  🤖 {question}")
        print(f"{'='*60}")

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
        graph.invoke(Command(resume=answer), config)

    # Check final state
    final_state = graph.get_state(config)
    values = final_state.values

    if values.get("trend_report"):
        print(f"\n{'='*60}")
        print(values["trend_report"])
        print(f"{'='*60}")

    return values.get("trend_report")


def start_interview(
    query: str,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Start a new trend interview session (programmatic API)."""
    graph = build_interview_graph()

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

    graph.invoke(initial_state, config)

    state = graph.get_state(config)
    interrupt_value = state.tasks[0].interrupts[0].value

    return {
        "thread_id": thread_id,
        "question": interrupt_value.get("question"),
        "round": interrupt_value.get("round", 0),
        "field": interrupt_value.get("field"),
    }


def resume_interview(
    thread_id: str,
    answer: str,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Resume a paused interview with an answer (programmatic API)."""
    # Note: MemorySaver doesn't persist across process boundaries.
    # For production, use SqliteSaver with proper context manager.
    graph = build_interview_graph()
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(Command(resume=answer), config)
    state = graph.get_state(config)

    response = {
        "thread_id": thread_id,
        "round": state.values.get("round", 0),
        "clarity_score": state.values.get("clarity_score", 0.0),
        "is_complete": state.values.get("is_complete", False),
        "topic": state.values.get("topic", ""),
        "requirements": state.values.get("requirements", []),
    }

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
