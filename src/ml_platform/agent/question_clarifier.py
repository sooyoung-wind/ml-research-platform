"""ML Research Platform — Question Clarifier.

Analyzes user questions to determine clarity and research scope.
Uses LLM to score question clarity (1-5) and suggest improvements.

If clarity < 3, triggers interactive interview to refine the query.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from ml_platform.config import config


class ClarityLevel(IntEnum):
    """Question clarity levels."""
    UNCLEAR = 1      # Too vague, need full interview
    VAGUE = 2        # Partially clear, need some clarification
    MODERATE = 3     # Clear enough to start, could be refined
    CLEAR = 4        # Well-defined, minor clarifications possible
    PRECISE = 5      # Extremely specific, ready to search


@dataclass
class ClarificationResult:
    """Result of question clarity analysis.

    Attributes:
        original_question: The user's original question.
        clarity_score: Clarity level (1-5).
        clarity_level: Human-readable clarity level name.
        detected_domain: Detected research domain (e.g. 'NLP', 'CV', 'RL').
        detected_task: Detected specific task (e.g. 'text generation', 'object detection').
        detected_method: Detected method class (e.g. 'transformer', 'diffusion').
        suggested_keywords: Search keywords extracted from the question.
        missing_info: List of aspects that need clarification.
        suggested_questions: Follow-up questions to ask the user.
        refined_query: Refined search query (if clarity >= 3).
        search_sources: Recommended search sources.
    """
    original_question: str = ""
    clarity_score: int = 1
    clarity_level: str = "unclear"
    detected_domain: str = ""
    detected_task: str = ""
    detected_method: str = ""
    suggested_keywords: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    suggested_questions: list[str] = field(default_factory=list)
    refined_query: str = ""
    search_sources: list[str] = field(default_factory=list)


class QuestionClarifier:
    """Analyzes user questions and determines research scope.

    Uses LLM to:
    1. Score question clarity (1-5)
    2. Detect domain/task/method
    3. Extract search keywords
    4. Identify missing information
    5. Generate follow-up questions if needed
    6. Produce a refined search query

    Usage:
        clarifier = QuestionClarifier()
        result = clarifier.analyze("RAG에서 hallucination 줄이는 최신 연구")
        if result.clarity_score < 3:
            # Need interview
            for q in result.suggested_questions:
                answer = ask_user(q)
        else:
            # Ready to search
            papers = search(result.refined_query)
    """

    SYSTEM_PROMPT = """You are a research query analyzer for an ML research platform.
Given a user's research question, analyze it and return a JSON object with:

1. "clarity_score" (1-5): How clear/specific the question is
   - 1: Too vague ("AI에 대해 알려줘")
   - 2: Partial ("RAG 관련 논문 찾아줘")
   - 3: Moderate ("RAG에서 hallucination 줄이는 방법")
   - 4: Clear ("RAG 시스템에서 retrieval accuracy를 높이는 최신 방법론")
   - 5: Precise ("2024년 이후 RAG 시스템에서 cross-encoder reranking과 query expansion을 결합한 논문")

2. "detected_domain": Primary research domain (NLP, CV, RL, ML, DL, etc.)
3. "detected_task": Specific task if mentioned (text generation, summarization, etc.)
4. "detected_method": Method class if mentioned (transformer, diffusion, GNN, etc.)
5. "suggested_keywords": 3-5 search keywords for paper search (English)
6. "missing_info": What information is missing or unclear (list of strings)
7. "suggested_questions": 2-3 follow-up questions to clarify (in Korean, matching user's language)
8. "refined_query": Refined English search query combining all detected intent
9. "search_sources": Recommended sources from ["arxiv", "semantic_scholar", "huggingface", "paperswithcode"]

Return ONLY valid JSON, no explanation."""

    def __init__(self) -> None:
        self.provider = os.getenv("ML_DEFAULT_LLM_PROVIDER", "ollama")
        self.model = os.getenv("ML_DEFAULT_LLM_MODEL", "gemma4:31b-cloud")

    def analyze(self, question: str) -> ClarificationResult:
        """Analyze a research question and return clarity assessment.

        Args:
            question: User's research question (any language).

        Returns:
            ClarificationResult with clarity score, keywords, and suggestions.
        """
        result = ClarificationResult(original_question=question)

        try:
            response = self._call_llm(question)
            parsed = self._parse_response(response)
            self._fill_result(result, parsed)
        except Exception as e:
            # Fallback: treat as moderate clarity with raw question as keyword
            result.clarity_score = 3
            result.clarity_level = "moderate"
            result.suggested_keywords = [question]
            result.refined_query = question
            result.search_sources = ["arxiv", "semantic_scholar"]
            result.missing_info = [f"LLM analysis failed: {e}"]

        return result

    def _call_llm(self, question: str) -> str:
        """Call LLM via litellm for question analysis."""
        from litellm import completion
        import litellm
        litellm.drop_params = True

        response = completion(
            model=f"{self.provider}/{self.model}",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM JSON response, handling markdown code blocks."""
        text = response.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Skip first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    def _fill_result(self, result: ClarificationResult, data: dict) -> None:
        """Fill ClarificationResult from parsed LLM response."""
        result.clarity_score = min(5, max(1, data.get("clarity_score", 3)))
        result.clarity_level = ClarityLevel(result.clarity_score).name.lower()
        result.detected_domain = data.get("detected_domain", "")
        result.detected_task = data.get("detected_task", "")
        result.detected_method = data.get("detected_method", "")
        result.suggested_keywords = data.get("suggested_keywords", [])
        result.missing_info = data.get("missing_info", [])
        result.suggested_questions = data.get("suggested_questions", [])
        result.refined_query = data.get("refined_query", result.original_question)
        result.search_sources = data.get("search_sources", ["arxiv", "semantic_scholar"])
