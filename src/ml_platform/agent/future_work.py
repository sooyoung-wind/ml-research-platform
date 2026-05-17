"""ML Research Platform — Future Work Planner.

Generates future work suggestions and improvement plans based on:
  1. Original paper's stated future work
  2. Reproducibility analysis results
  3. Cross-paper synthesis (common gaps, open problems)

Output includes:
  - Prioritized research directions
  - Specific improvement tasks
  - Estimated difficulty levels
  - Suggested experiments
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FutureWorkItem:
    """A single future work suggestion.

    Attributes:
        title: Short title of the suggested work.
        description: Detailed description.
        priority: Priority level (high, medium, low).
        difficulty: Estimated difficulty (easy, moderate, hard).
        related_papers: Paper IDs this builds on.
        category: Category (improvement, extension, novel, evaluation).
        suggested_experiments: Specific experiments to try.
    """
    title: str = ""
    description: str = ""
    priority: str = "medium"
    difficulty: str = "moderate"
    related_papers: list[str] = field(default_factory=list)
    category: str = "improvement"
    suggested_experiments: list[str] = field(default_factory=list)


@dataclass
class FutureWorkReport:
    """Complete future work analysis.

    Attributes:
        session_id: Session identifier.
        question: Original research question.
        items: Prioritized future work suggestions.
        summary: Executive summary.
        synthesized_gaps: Common gaps across papers.
    """
    session_id: str = ""
    question: str = ""
    items: list[FutureWorkItem] = field(default_factory=list)
    summary: str = ""
    synthesized_gaps: list[str] = field(default_factory=list)


class FutureWorkPlanner:
    """Plans future research directions based on analysis results.

    Usage:
        planner = FutureWorkPlanner()
        report = planner.plan(
            question="RAG hallucination",
            papers=[...],
            reproducibility_reports=[...],
            kg_stats={...},
        )
    """

    PLANNING_PROMPT = """Based on the following research analysis, generate a prioritized future work plan.

Original question: {question}

Analyzed papers:
{paper_summaries}

Reproducibility results:
{repro_results}

Knowledge graph insights:
{kg_insights}

Generate 5-8 future work items as JSON:
{{
    "summary": "executive summary of research landscape",
    "synthesized_gaps": ["gap 1", "gap 2"],
    "items": [
        {{
            "title": "short title",
            "description": "detailed description",
            "priority": "high|medium|low",
            "difficulty": "easy|moderate|hard",
            "related_papers": ["paper_id_1"],
            "category": "improvement|extension|novel|evaluation",
            "suggested_experiments": ["experiment 1"]
        }}
    ]
}}"""

    def __init__(self) -> None:
        self.provider = os.getenv("ML_DEFAULT_LLM_PROVIDER", "ollama")
        self.model = os.getenv("ML_DEFAULT_LLM_MODEL", "gemma4:31b-cloud")

    def plan(
        self,
        question: str,
        papers: list[dict],
        reproducibility_reports: list[dict] | None = None,
        kg_stats: dict | None = None,
        session_id: str = "",
    ) -> FutureWorkReport:
        """Generate future work plan.

        Args:
            question: Original research question.
            papers: List of paper dicts.
            reproducibility_reports: Reproducibility analysis results.
            kg_stats: Knowledge graph statistics.
            session_id: Session identifier.

        Returns:
            FutureWorkReport with prioritized suggestions.
        """
        report = FutureWorkReport(
            session_id=session_id,
            question=question,
        )

        # Build summaries
        paper_summaries = "\n".join(
            f"- {p.get('arxiv_id', '?')}: {p.get('title', '?')[:80]}"
            for p in papers[:15]
        )

        repro_text = ""
        if reproducibility_reports:
            repro_text = "\n".join(
                f"- {r.get('paper_id', '?')}: score={r.get('overall_score', 0)}%, "
                f"status={r.get('status', 'unknown')}"
                for r in reproducibility_reports
            )

        kg_text = f"Nodes: {kg_stats.get('nodes', 0)}, Edges: {kg_stats.get('edges', 0)}" if kg_stats else "N/A"

        try:
            from litellm import completion
            import litellm
            litellm.drop_params = True

            prompt = self.PLANNING_PROMPT.format(
                question=question,
                paper_summaries=paper_summaries,
                repro_results=repro_text or "No reproduction attempted",
                kg_insights=kg_text,
            )
            response = completion(
                model=f"{self.provider}/{self.model}",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            text = response.choices[0].message.content or "{}"
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = {}

            report.summary = data.get("summary", "")
            report.synthesized_gaps = data.get("synthesized_gaps", [])

            for item in data.get("items", []):
                report.items.append(FutureWorkItem(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    priority=item.get("priority", "medium"),
                    difficulty=item.get("difficulty", "moderate"),
                    related_papers=item.get("related_papers", []),
                    category=item.get("category", "improvement"),
                    suggested_experiments=item.get("suggested_experiments", []),
                ))

        except Exception as e:
            report.summary = f"LLM planning failed: {e}"

        return report
