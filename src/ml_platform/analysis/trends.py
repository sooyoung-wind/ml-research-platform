"""ML Research Platform — Trend Analysis module.

Analyzes research trends across papers in the database including
temporal topic trends, methodology comparisons, and research gap detection.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ml_platform.db import PapersDB
from ml_platform.models import Paper, ProcessingStatus


# ── Data models ──────────────────────────────────────────────────────


class YearBucket(BaseModel):
    """Papers counted in a single year.

    Attributes:
        year: Publication year.
        count: Number of papers.
        papers: List of paper IDs.
    """
    year: int
    count: int
    papers: list[str] = Field(default_factory=list)


class TopicTrend(BaseModel):
    """Trend data for a single topic/category.

    Attributes:
        topic: Topic name.
        total: Total paper count.
        yearly: Year-by-year breakdown.
        growth_rate: Growth rate from previous period (0.0 = stable).
    """
    topic: str
    total: int = 0
    yearly: list[YearBucket] = Field(default_factory=list)
    growth_rate: float = 0.0


class MethodComparison(BaseModel):
    """Comparison of a methodology across papers.

    Attributes:
        method_name: Method name.
        papers_using: Number of papers using this method.
        paper_ids: Paper IDs.
        domains: Domains where this method is applied.
        first_seen: Year first observed.
        latest_seen: Year most recently observed.
    """
    method_name: str
    papers_using: int = 0
    paper_ids: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    first_seen: int | None = None
    latest_seen: int | None = None


class ResearchGap(BaseModel):
    """Identified research gap.

    Attributes:
        gap_type: Type of gap (underexplored, emerging, declining).
        description: Description of the gap.
        related_topics: Topics related to this gap.
        opportunity_score: Score 0-1 indicating opportunity level.
        papers_count: Number of papers in this area.
    """
    gap_type: str  # underexplored, emerging, declining
    description: str
    related_topics: list[str] = Field(default_factory=list)
    opportunity_score: float = 0.0
    papers_count: int = 0


class TrendReport(BaseModel):
    """Complete trend analysis report.

    Attributes:
        total_papers: Total papers analyzed.
        year_range: (earliest_year, latest_year).
        top_topics: Top trending topics.
        declining_topics: Declining topics.
        method_trends: Method comparison data.
        research_gaps: Identified research gaps.
        generated_at: Report generation timestamp.
    """
    total_papers: int = 0
    year_range: tuple[int, int] = (0, 0)
    top_topics: list[TopicTrend] = Field(default_factory=list)
    declining_topics: list[TopicTrend] = Field(default_factory=list)
    method_trends: list[MethodComparison] = Field(default_factory=list)
    research_gaps: list[ResearchGap] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


# ── Trend Analyzer ───────────────────────────────────────────────────


class TrendAnalyzer:
    """Analyzes research trends across papers in the database.

    Provides temporal topic analysis, methodology comparison,
    and research gap detection from paper metadata and analyses.

    Usage::

        analyzer = TrendAnalyzer()
        report = analyzer.analyze()
        for trend in report.top_topics:
            print(f"{trend.topic}: {trend.total} papers, growth={trend.growth_rate:.1%}")
    """

    def __init__(self, db: PapersDB | None = None) -> None:
        """Initialize the analyzer.

        Args:
            db: PapersDB instance. Creates a default one if not provided.
        """
        self.db = db or PapersDB()

    def analyze(
        self,
        papers: list[Paper] | None = None,
        top_n: int = 10,
    ) -> TrendReport:
        """Run a complete trend analysis.

        Args:
            papers: Optional list of papers. Uses all DB papers if None.
            top_n: Number of top/declining topics to return.

        Returns:
            TrendReport with all analysis results.
        """
        start = time.time()

        if papers is None:
            papers = self.db.get_papers(limit=1000)

        if not papers:
            return TrendReport()

        print(f"[INFO] Analyzing trends across {len(papers)} papers...")

        # Year range
        years = [p.year for p in papers if p.year]
        year_range = (min(years), max(years)) if years else (0, 0)

        # Topic trends
        topic_trends = self._compute_topic_trends(papers)
        sorted_trends = sorted(topic_trends, key=lambda t: t.growth_rate, reverse=True)
        top_topics = sorted_trends[:top_n]
        declining_topics = sorted_trends[-top_n:] if len(sorted_trends) > top_n else []

        # Method trends (from analyses)
        method_trends = self._compute_method_trends(papers)

        # Research gaps
        gaps = self._detect_research_gaps(papers, topic_trends)

        elapsed = time.time() - start
        print(f"[INFO] Trend analysis complete ({elapsed:.1f}s)")

        return TrendReport(
            total_papers=len(papers),
            year_range=year_range,
            top_topics=top_topics,
            declining_topics=declining_topics,
            method_trends=method_trends,
            research_gaps=gaps,
        )

    def _compute_topic_trends(self, papers: list[Paper]) -> list[TopicTrend]:
        """Compute trends by arXiv category and extracted topics.

        Args:
            papers: List of papers.

        Returns:
            List of TopicTrend objects.
        """
        # Group by category
        cat_by_year: dict[str, dict[int, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for paper in papers:
            if not paper.year:
                continue
            for cat in paper.categories:
                cat_by_year[cat][paper.year].append(paper.paper_id)

        trends: list[TopicTrend] = []
        for cat, year_data in cat_by_year.items():
            yearly = [
                YearBucket(year=y, count=len(ids), papers=ids)
                for y, ids in sorted(year_data.items())
            ]
            total = sum(b.count for b in yearly)

            # Growth rate: compare last 2 years
            growth = 0.0
            if len(yearly) >= 2:
                recent = yearly[-1].count
                prev = yearly[-2].count
                if prev > 0:
                    growth = (recent - prev) / prev

            trends.append(TopicTrend(
                topic=cat,
                total=total,
                yearly=yearly,
                growth_rate=growth,
            ))

        return trends

    def _compute_method_trends(self, papers: list[Paper]) -> list[MethodComparison]:
        """Compute methodology trends from paper analyses.

        Args:
            papers: List of papers with potential analyses.

        Returns:
            List of MethodComparison objects.
        """
        methods: dict[str, dict] = defaultdict(lambda: {
            "paper_ids": [],
            "domains": set(),
            "years": set(),
        })

        for paper in papers:
            analysis = self.db.get_analysis_object(paper.paper_id, paper.source.value)
            if analysis is None:
                continue

            how_text = analysis.five_w1h.how
            if not how_text:
                continue

            # Extract method mentions using entity_resolver patterns
            from ml_platform.graph.entity_resolver import _extract_methods
            method_nodes = []
            _extract_methods(how_text, method_nodes)

            for node in method_nodes:
                name = node.label
                methods[name]["paper_ids"].append(paper.paper_id)
                if analysis.domain:
                    methods[name]["domains"].add(analysis.domain)
                if paper.year:
                    methods[name]["years"].add(paper.year)

        comparisons = []
        for name, data in methods.items():
            years = sorted(data["years"]) if data["years"] else []
            comparisons.append(MethodComparison(
                method_name=name,
                papers_using=len(data["paper_ids"]),
                paper_ids=data["paper_ids"],
                domains=sorted(data["domains"]),
                first_seen=years[0] if years else None,
                latest_seen=years[-1] if years else None,
            ))

        return sorted(comparisons, key=lambda m: m.papers_using, reverse=True)

    def _detect_research_gaps(
        self,
        papers: list[Paper],
        topic_trends: list[TopicTrend],
    ) -> list[ResearchGap]:
        """Detect research gaps based on topic trends.

        Args:
            papers: List of papers.
            topic_trends: Computed topic trends.

        Returns:
            List of ResearchGap objects.
        """
        gaps: list[ResearchGap] = []

        # Category co-occurrence to find underexplored combinations
        cat_pairs: Counter = Counter()
        paper_cats: dict[str, set[str]] = {}
        for paper in papers:
            cats = set(paper.categories)
            paper_cats[paper.paper_id] = cats
            # Count all pairs
            for c1 in cats:
                for c2 in cats:
                    if c1 < c2:
                        cat_pairs[(c1, c2)] += 1

        # 1. Underexplored: categories with very few papers
        for trend in topic_trends:
            if trend.total <= 2 and trend.growth_rate <= 0:
                gaps.append(ResearchGap(
                    gap_type="underexplored",
                    description=f"Category '{trend.topic}' has only {trend.total} paper(s) "
                                f"and declining/stable growth ({trend.growth_rate:.0%}).",
                    related_topics=[trend.topic],
                    opportunity_score=0.7 if trend.total == 1 else 0.5,
                    papers_count=trend.total,
                ))

        # 2. Emerging: high growth rate
        for trend in topic_trends:
            if trend.growth_rate >= 0.5 and trend.total >= 2:
                gaps.append(ResearchGap(
                    gap_type="emerging",
                    description=f"Category '{trend.topic}' shows strong growth "
                                f"({trend.growth_rate:.0%}) with {trend.total} papers.",
                    related_topics=[trend.topic],
                    opportunity_score=min(0.9, 0.5 + trend.growth_rate * 0.3),
                    papers_count=trend.total,
                ))

        # 3. Declining: negative growth
        for trend in topic_trends:
            if trend.growth_rate < -0.3 and trend.total >= 3:
                gaps.append(ResearchGap(
                    gap_type="declining",
                    description=f"Category '{trend.topic}' is declining "
                                f"({trend.growth_rate:.0%}) despite having {trend.total} papers.",
                    related_topics=[trend.topic],
                    opportunity_score=0.3,
                    papers_count=trend.total,
                ))

        # Sort by opportunity score
        gaps.sort(key=lambda g: g.opportunity_score, reverse=True)
        return gaps[:20]

    def compute_yearly_summary(self, papers: list[Paper] | None = None) -> list[dict]:
        """Compute yearly publication summary.

        Args:
            papers: Optional list of papers.

        Returns:
            List of dicts with year, count, top_categories.
        """
        if papers is None:
            papers = self.db.get_papers(limit=1000)

        by_year: dict[int, dict] = defaultdict(lambda: {
            "count": 0,
            "categories": Counter(),
            "paper_ids": [],
        })

        for paper in papers:
            if not paper.year:
                continue
            by_year[paper.year]["count"] += 1
            by_year[paper.year]["paper_ids"].append(paper.paper_id)
            for cat in paper.categories:
                by_year[paper.year]["categories"][cat] += 1

        result = []
        for year in sorted(by_year.keys()):
            data = by_year[year]
            top_cats = data["categories"].most_common(5)
            result.append({
                "year": year,
                "count": data["count"],
                "top_categories": [
                    {"category": cat, "count": cnt} for cat, cnt in top_cats
                ],
                "paper_ids": data["paper_ids"],
            })

        return result

    def compute_category_matrix(self, papers: list[Paper] | None = None) -> dict:
        """Compute category co-occurrence matrix.

        Args:
            papers: Optional list of papers.

        Returns:
            Dict with categories and matrix data.
        """
        if papers is None:
            papers = self.db.get_papers(limit=1000)

        all_cats: set[str] = set()
        for paper in papers:
            all_cats.update(paper.categories)
        all_cats = sorted(all_cats)

        # Build co-occurrence matrix
        matrix: dict[str, dict[str, int]] = {
            c1: {c2: 0 for c2 in all_cats} for c1 in all_cats
        }
        for paper in papers:
            for c1 in paper.categories:
                for c2 in paper.categories:
                    matrix[c1][c2] += 1

        return {
            "categories": all_cats,
            "matrix": matrix,
        }

    def generate_report_markdown(self, report: TrendReport) -> str:
        """Generate a markdown trend report.

        Args:
            report: TrendReport to format.

        Returns:
            Markdown string.
        """
        lines = [
            "# Research Trend Report",
            f"",
            f"**Papers analyzed:** {report.total_papers}",
            f"**Year range:** {report.year_range[0]}–{report.year_range[1]}",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M')}",
            f"",
        ]

        # Top trending
        if report.top_topics:
            lines.append("## Trending Topics")
            lines.append("")
            lines.append("| Topic | Papers | Growth | Yearly Trend |")
            lines.append("|-------|--------|--------|-------------|")
            for t in report.top_topics:
                yearly_str = ", ".join(
                    f"{b.year}:{b.count}" for b in t.yearly[-4:]
                )
                growth = f"{t.growth_rate:+.0%}" if t.growth_rate != 0 else "—"
                lines.append(f"| {t.topic} | {t.total} | {growth} | {yearly_str} |")
            lines.append("")

        # Declining
        if report.declining_topics:
            lines.append("## Declining Topics")
            lines.append("")
            for t in report.declining_topics:
                if t.growth_rate < 0:
                    growth = f"{t.growth_rate:.0%}"
                    lines.append(f"- **{t.topic}**: {t.total} papers, growth {growth}")
            lines.append("")

        # Method trends
        if report.method_trends:
            lines.append("## Methodology Trends")
            lines.append("")
            lines.append("| Method | Papers | Domains | Period |")
            lines.append("|--------|--------|---------|--------|")
            for m in report.method_trends:
                period = f"{m.first_seen}–{m.latest_seen}" if m.first_seen else "—"
                domains = ", ".join(m.domains[:3]) if m.domains else "—"
                lines.append(f"| {m.method_name} | {m.papers_using} | {domains} | {period} |")
            lines.append("")

        # Research gaps
        if report.research_gaps:
            lines.append("## Research Gaps")
            lines.append("")
            for g in report.research_gaps:
                icon = {
                    "underexplored": "🕳️",
                    "emerging": "🌱",
                    "declining": "📉",
                }.get(g.gap_type, "❓")
                score_bar = "█" * int(g.opportunity_score * 10)
                lines.append(
                    f"### {icon} [{g.gap_type.title()}] "
                    f"Opportunity: {g.opportunity_score:.0%} {score_bar}"
                )
                lines.append(f"{g.description}")
                lines.append(f"Papers in area: {g.papers_count}")
                lines.append(f"Related: {', '.join(g.related_topics)}")
                lines.append("")

        return "\n".join(lines)

    def generate_report_html(self, report: TrendReport) -> str:
        """Generate a standalone HTML trend report.

        Args:
            report: TrendReport to format.

        Returns:
            HTML string.
        """
        yearly = self.compute_yearly_summary()

        # Build yearly chart data
        chart_labels = [str(item["year"]) for item in yearly]
        chart_data = [item["count"] for item in yearly]

        # Category breakdown
        cat_data: dict[str, list[int]] = defaultdict(
            lambda: [0] * len(yearly)
        )
        for i, item in enumerate(yearly):
            for cat_info in item["top_categories"]:
                cat_data[cat_info["category"]][i] = cat_info["count"]

        # Top categories chart (top 8 by total)
        cat_totals = {
            cat: sum(counts) for cat, counts in cat_data.items()
        }
        top_cats = sorted(cat_totals.keys(), key=lambda c: cat_totals.get(c, 0), reverse=True)[:8]

        top_cats_json = json.dumps(top_cats)
        cat_series = []
        for cat in top_cats:
            cat_series.append({
                "label": cat,
                "data": cat_data[cat],
                "backgroundColor": _cat_color(cat),
            })
        cat_series_json = json.dumps(cat_series)

        # Trending topics table
        trending_rows = ""
        for t in report.top_topics[:15]:
            yearly_str = " → ".join(
                f'<span title="{b.year}: {b.count} papers">{b.count}</span>'
                for b in t.yearly[-5:]
            )
            growth_class = "positive" if t.growth_rate > 0 else "negative" if t.growth_rate < 0 else ""
            growth_str = f"{t.growth_rate:+.0%}" if t.growth_rate != 0 else "—"
            trending_rows += f"""
            <tr>
                <td><strong>{t.topic}</strong></td>
                <td>{t.total}</td>
                <td class="{growth_class}">{growth_str}</td>
                <td>{yearly_str}</td>
            </tr>"""

        # Research gaps
        gaps_html = ""
        for g in report.research_gaps[:10]:
            icon = {"underexplored": "🕳️", "emerging": "🌱", "declining": "📉"}.get(
                g.gap_type, "❓"
            )
            bar_width = int(g.opportunity_score * 100)
            gaps_html += f"""
            <div class="gap-card {g.gap_type}">
                <div class="gap-header">
                    <span class="gap-icon">{icon}</span>
                    <span class="gap-type">{g.gap_type.title()}</span>
                    <span class="gap-score">{g.opportunity_score:.0%}</span>
                </div>
                <div class="gap-bar"><div class="gap-fill" style="width:{bar_width}%"></div></div>
                <p class="gap-desc">{g.description}</p>
                <p class="gap-meta">Papers: {g.papers_count} | Related: {', '.join(g.related_topics)}</p>
            </div>"""

        # Method trends table
        method_rows = ""
        for m in report.method_trends[:10]:
            period = f"{m.first_seen}–{m.latest_seen}" if m.first_seen else "—"
            domains = ", ".join(m.domains[:3]) if m.domains else "—"
            method_rows += f"""
            <tr>
                <td><strong>{m.method_name}</strong></td>
                <td>{m.papers_using}</td>
                <td>{domains}</td>
                <td>{period}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Trend Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f0f1a;
    color: #e0e0e0;
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
  }}
  h1 {{ color: #4ECDC4; margin-bottom: 5px; font-size: 28px; }}
  h2 {{ color: #4ECDC4; margin: 30px 0 15px; font-size: 22px; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  .meta {{ color: #888; font-size: 14px; margin-bottom: 30px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
  .chart-box {{ background: #1a1a2e; border-radius: 12px; padding: 20px; }}
  .chart-box h3 {{ color: #aaa; font-size: 14px; margin-bottom: 10px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
  th {{ text-align: left; padding: 10px; border-bottom: 2px solid #333; color: #888; font-size: 13px; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #222; font-size: 14px; }}
  tr:hover {{ background: #1a1a2e; }}
  .positive {{ color: #4ECDC4; font-weight: bold; }}
  .negative {{ color: #FF6B6B; font-weight: bold; }}
  .gap-card {{
    background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 10px 0;
    border-left: 4px solid #45B7D1;
  }}
  .gap-card.underexplored {{ border-left-color: #FFEAA7; }}
  .gap-card.emerging {{ border-left-color: #4ECDC4; }}
  .gap-card.declining {{ border-left-color: #FF6B6B; }}
  .gap-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
  .gap-type {{ font-weight: bold; text-transform: uppercase; font-size: 12px; color: #888; }}
  .gap-score {{ margin-left: auto; font-weight: bold; color: #4ECDC4; }}
  .gap-bar {{ height: 4px; background: #333; border-radius: 2px; margin: 8px 0; }}
  .gap-fill {{ height: 100%; background: linear-gradient(90deg, #4ECDC4, #45B7D1); border-radius: 2px; }}
  .gap-desc {{ font-size: 14px; line-height: 1.5; }}
  .gap-meta {{ font-size: 12px; color: #666; margin-top: 5px; }}
  @media (max-width: 768px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
  <h1>Research Trend Report</h1>
  <p class="meta">
    {report.total_papers} papers | {report.year_range[0]}–{report.year_range[1]} |
    Generated {report.generated_at.strftime('%Y-%m-%d %H:%M')}
  </p>

  <div class="charts">
    <div class="chart-box">
      <h3>Publications by Year</h3>
      <canvas id="yearChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Top Categories Over Time</h3>
      <canvas id="catChart"></canvas>
    </div>
  </div>

  <h2>Trending Topics</h2>
  <table>
    <tr><th>Category</th><th>Papers</th><th>Growth</th><th>Yearly Trend</th></tr>
    {trending_rows}
  </table>

  <h2>Methodology Trends</h2>
  {"<table><tr><th>Method</th><th>Papers</th><th>Domains</th><th>Period</th></tr>" + method_rows + "</table>" if method_rows else "<p>No method analyses available yet. Run 'ml-research analyze paper' first.</p>"}

  <h2>Research Gaps</h2>
  {gaps_html if gaps_html else "<p>No significant gaps detected.</p>"}

  <script>
    // Year chart
    new Chart(document.getElementById('yearChart'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(chart_labels)},
        datasets: [{{
          label: 'Papers',
          data: {json.dumps(chart_data)},
          backgroundColor: '#4ECDC4',
          borderRadius: 6,
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          y: {{ beginAtZero: true, grid: {{ color: '#222' }}, ticks: {{ color: '#888' }} }},
          x: {{ grid: {{ display: false }}, ticks: {{ color: '#888' }} }}
        }}
      }}
    }});

    // Category stacked chart
    new Chart(document.getElementById('catChart'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(chart_labels)},
        datasets: {cat_series_json}
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#888', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ stacked: true, grid: {{ display: false }}, ticks: {{ color: '#888' }} }},
          y: {{ stacked: true, grid: {{ color: '#222' }}, ticks: {{ color: '#888' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>"""
        return html


def _cat_color(cat: str) -> str:
    """Assign a consistent color to a category.

    Args:
        cat: arXiv category string.

    Returns:
        Hex color string.
    """
    colors = [
        "#4ECDC4", "#FF6B6B", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    ]
    return colors[hash(cat) % len(colors)]
