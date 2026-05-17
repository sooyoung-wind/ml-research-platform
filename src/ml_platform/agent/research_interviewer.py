"""ML Research Platform — Research Interviewer.

Interactive multi-round interview to clarify research intent.
Generalized from analysis/interview.py for use across all research flows.

Flow:
  1. QuestionClarifier analyzes the initial question
  2. If clarity < threshold, start interview rounds
  3. Each round: LLM generates question → user answers → update context
  4. When clarity reaches threshold or max rounds hit → produce SearchStrategy
  5. SearchStrategy drives multi-source paper collection
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from ml_platform.agent.question_clarifier import ClarificationResult, QuestionClarifier


@dataclass
class SearchStrategy:
    """Concrete search plan derived from interview.

    Attributes:
        original_question: User's original question.
        refined_query: Refined English search query.
        keywords: Search keywords for each source.
        domains: Target research domains.
        methods: Specific methods/techniques to focus on.
        sources: Which sources to search.
        year_range: (start_year, end_year) or None for all.
        max_papers_per_source: Max papers from each source.
        focus_areas: Specific aspects to prioritize.
        exclusion_terms: Terms to exclude from search.
    """
    original_question: str = ""
    refined_query: str = ""
    keywords: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=lambda: ["arxiv", "semantic_scholar"])
    year_range: tuple[int, int] | None = None
    max_papers_per_source: int = 10
    focus_areas: list[str] = field(default_factory=list)
    exclusion_terms: list[str] = field(default_factory=list)


@dataclass
class InterviewRound:
    """Single interview round.

    Attributes:
        round_number: Round number (1-indexed).
        question: Question asked to user.
        answer: User's answer (empty until answered).
        extracted_info: Key info extracted from answer.
    """
    round_number: int = 0
    question: str = ""
    answer: str = ""
    extracted_info: list[str] = field(default_factory=list)


@dataclass
class InterviewState:
    """State of an ongoing interview.

    Attributes:
        original_question: The initial question.
        rounds: Completed interview rounds.
        accumulated_context: All extracted info so far.
        current_clarity: Latest clarity score.
        is_complete: Whether interview is done.
        final_strategy: The derived SearchStrategy (if complete).
    """
    original_question: str = ""
    rounds: list[InterviewRound] = field(default_factory=list)
    accumulated_context: list[str] = field(default_factory=list)
    current_clarity: int = 1
    is_complete: bool = False
    final_strategy: SearchStrategy | None = None


class ResearchInterviewer:
    """Multi-round interview to refine research queries.

    Usage:
        interviewer = ResearchInterviewer()
        state = interviewer.start("RAG 관련 연구")

        while not state.is_complete:
            print(interviewer.get_next_question(state))
            answer = input("> ")
            interviewer.process_answer(state, answer)

        strategy = state.final_strategy
    """

    INTERVIEW_PROMPT = """You are a research interview assistant. Based on the user's research question and previous interview context, generate ONE follow-up question to better understand their research needs.

Rules:
- Ask in Korean (matching user's language)
- Be specific and actionable
- Focus on: specific problem, method preference, dataset/domain constraints, timeline expectations
- If the user's answer is comprehensive enough, indicate completion

Context so far:
- Original question: {question}
- Accumulated info: {context}
- Current clarity: {clarity}/5

Return JSON:
{{
    "question": "your follow-up question in Korean",
    "is_sufficient": true/false,
    "extracted_keywords": ["keyword1", "keyword2"],
    "updated_search_query": "refined English query",
    "suggested_sources": ["arxiv", "semantic_scholar", ...],
    "focus_areas": ["area1", "area2"],
    "year_range": [2020, 2025] or null
}}"""

    STRATEGY_PROMPT = """Based on the completed interview, generate a comprehensive search strategy.

Original question: {question}
Accumulated context: {context}
Keywords: {keywords}

Return JSON:
{{
    "refined_query": "final English search query",
    "keywords": ["k1", "k2", "k3", "k4", "k5"],
    "domains": ["domain1", "domain2"],
    "methods": ["method1"],
    "sources": ["arxiv", "semantic_scholar", "huggingface", "paperswithcode"],
    "year_range": [start_year, end_year] or null,
    "max_papers_per_source": 10,
    "focus_areas": ["area1", "area2"],
    "exclusion_terms": ["term1"]
}}"""

    def __init__(
        self,
        min_clarity: int = 3,
        max_rounds: int = 5,
        min_rounds: int = 2,
    ) -> None:
        self.min_clarity = min_clarity
        self.max_rounds = max_rounds
        self.min_rounds = min_rounds
        self.provider = os.getenv("ML_DEFAULT_LLM_PROVIDER", "ollama")
        self.model = os.getenv("ML_DEFAULT_LLM_MODEL", "gemma4:31b-cloud")
        self._clarifier = QuestionClarifier()

    def start(self, question: str) -> InterviewState:
        """Start a new interview session.

        First analyzes the question, then prepares the initial state.
        """
        state = InterviewState(original_question=question)

        # Initial clarity check
        clarification = self._clarifier.analyze(question)
        state.current_clarity = clarification.clarity_score

        # Store initial context
        if clarification.detected_domain:
            state.accumulated_context.append(f"Domain: {clarification.detected_domain}")
        if clarification.detected_task:
            state.accumulated_context.append(f"Task: {clarification.detected_task}")
        if clarification.detected_method:
            state.accumulated_context.append(f"Method: {clarification.detected_method}")
        if clarification.suggested_keywords:
            state.accumulated_context.append(f"Keywords: {', '.join(clarification.suggested_keywords)}")

        # If already clear enough and minimum rounds met, skip interview
        if state.current_clarity >= self.min_clarity:
            state.is_complete = True
            state.final_strategy = self._build_strategy_from_clarification(
                question, clarification
            )
            return state

        return state

    def get_next_question(self, state: InterviewState) -> str | None:
        """Get the next interview question, or None if interview is complete."""
        if state.is_complete:
            return None

        if len(state.rounds) >= self.max_rounds:
            # Max rounds reached, finalize
            state.is_complete = True
            state.final_strategy = self._build_strategy_from_state(state)
            return None

        if len(state.rounds) == 0:
            # First round: use clarifier's suggested questions
            clarification = self._clarifier.analyze(state.original_question)
            if clarification.suggested_questions:
                return clarification.suggested_questions[0]

        # Generate question via LLM
        try:
            return self._generate_question(state)
        except Exception:
            # Fallback questions
            defaults = [
                "어떤 구체적인 문제를 해결하고 싶으신가요?",
                "특정 방법론이나 기술에 선호가 있으신가요?",
                "최근 몇 년 내의 연구에 관심이 있으신가요?",
                "어떤 형태의 결과를 기대하시나요? (논문 리뷰, 코드 구현, 비교 분석 등)",
                "특정 데이터셋이나 응용 분야가 있으신가요?",
            ]
            idx = min(len(state.rounds), len(defaults) - 1)
            return defaults[idx]

    def process_answer(self, state: InterviewState, answer: str) -> None:
        """Process user's answer and update state."""
        if not state.rounds or state.rounds[-1].answer:
            # Create new round
            round_num = len(state.rounds) + 1
            question = self.get_next_question(state) or "Additional context?"
            state.rounds.append(InterviewRound(
                round_number=round_num,
                question=question,
            ))

        current_round = state.rounds[-1]
        current_round.answer = answer

        # Extract info from answer
        try:
            extracted = self._extract_info(state, answer)
            current_round.extracted_info = extracted.get("extracted_keywords", [])
            state.accumulated_context.append(f"Round {current_round.round_number}: {answer}")

            # Update clarity
            is_sufficient = extracted.get("is_sufficient", False)
            if is_sufficient or len(state.rounds) >= self.min_rounds:
                state.current_clarity = min(5, state.current_clarity + 1)

            # Check if we can finalize
            if (state.current_clarity >= self.min_clarity and
                    len(state.rounds) >= self.min_rounds) or \
               len(state.rounds) >= self.max_rounds:
                state.is_complete = True
                state.final_strategy = self._build_strategy_from_state(state)
        except Exception:
            state.accumulated_context.append(f"Round {current_round.round_number}: {answer}")
            if len(state.rounds) >= self.min_rounds:
                state.current_clarity = min(5, state.current_clarity + 1)

    def _generate_question(self, state: InterviewState) -> str:
        """Generate next interview question via LLM."""
        from litellm import completion
        import litellm
        litellm.drop_params = True

        prompt = self.INTERVIEW_PROMPT.format(
            question=state.original_question,
            context="; ".join(state.accumulated_context[-5:]),
            clarity=state.current_clarity,
        )
        response = completion(
            model=f"{self.provider}/{self.model}",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("question", "추가로 알고 싶은 내용이 있으신가요?")

    def _extract_info(self, state: InterviewState, answer: str) -> dict:
        """Extract structured info from user's answer."""
        from litellm import completion
        import litellm
        litellm.drop_params = True

        prompt = self.INTERVIEW_PROMPT.format(
            question=state.original_question,
            context="; ".join(state.accumulated_context[-5:]) + f"; Latest answer: {answer}",
            clarity=state.current_clarity,
        )
        response = completion(
            model=f"{self.provider}/{self.model}",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        return json.loads(response.choices[0].message.content)

    def _build_strategy_from_clarification(
        self, question: str, c: ClarificationResult
    ) -> SearchStrategy:
        """Build SearchStrategy directly from clarification (no interview needed)."""
        return SearchStrategy(
            original_question=question,
            refined_query=c.refined_query,
            keywords=c.suggested_keywords,
            domains=[c.detected_domain] if c.detected_domain else [],
            methods=[c.detected_method] if c.detected_method else [],
            sources=c.search_sources,
            focus_areas=[],
        )

    def _build_strategy_from_state(self, state: InterviewState) -> SearchStrategy:
        """Build SearchStrategy from completed interview state."""
        try:
            from litellm import completion
            import litellm
            litellm.drop_params = True

            prompt = self.STRATEGY_PROMPT.format(
                question=state.original_question,
                context="; ".join(state.accumulated_context),
                keywords=", ".join(state.accumulated_context),
            )
            response = completion(
                model=f"{self.provider}/{self.model}",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            data = json.loads(response.choices[0].message.content)
            return SearchStrategy(
                original_question=state.original_question,
                refined_query=data.get("refined_query", state.original_question),
                keywords=data.get("keywords", []),
                domains=data.get("domains", []),
                methods=data.get("methods", []),
                sources=data.get("sources", ["arxiv", "semantic_scholar"]),
                year_range=tuple(data["year_range"]) if data.get("year_range") else None,
                max_papers_per_source=data.get("max_papers_per_source", 10),
                focus_areas=data.get("focus_areas", []),
                exclusion_terms=data.get("exclusion_terms", []),
            )
        except Exception:
            # Fallback: basic strategy from accumulated context
            return SearchStrategy(
                original_question=state.original_question,
                refined_query=state.original_question,
                keywords=state.accumulated_context,
                sources=["arxiv", "semantic_scholar"],
            )
