"""ML Research Platform — Research Dashboard Generator.

Generates an interactive HTML dashboard (Tailwind CSS) that:
  1. Shows collected papers with metadata
  2. Displays KG visualization
  3. Includes trend analysis
  4. Provides paper selection checkboxes
  5. Shows search strategy and interview summary

The dashboard is a self-contained HTML file that can be:
  - Opened directly in a browser
  - Served via a local HTTP server for selection submission
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ml_platform.config import config


DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="ko" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Dashboard — {title}</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body {{ background: #0f172a; color: #e2e8f0; font-family: 'Inter', system-ui, sans-serif; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; }}
  .paper-card {{ transition: all 0.2s; cursor: pointer; }}
  .paper-card:hover {{ border-color: #60a5fa; transform: translateY(-2px); }}
  .paper-card.selected {{ border-color: #34d399; background: #064e3b22; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; }}
  .source-arxiv {{ background: #b91c1c33; color: #fca5a5; }}
  .source-semantic_scholar {{ background: #1d4ed833; color: #93c5fd; }}
  .source-huggingface {{ background: #d9770633; color: #fcd34d; }}
  .source-paperswithcode {{ background: #7c3aed33; color: #c4b5fd; }}
  .stat-number {{ font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  #selection-bar {{ position: fixed; bottom: 0; left: 0; right: 0; background: #1e293b; border-top: 2px solid #334155; padding: 12px 24px; display: none; z-index: 50; }}
  .fade-in {{ animation: fadeIn 0.3s ease-in; }}
  @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
</style>
</head>
<body class="min-h-screen">

<!-- Header -->
<header class="border-b border-slate-700 px-6 py-4">
  <div class="max-w-7xl mx-auto flex items-center justify-between">
    <div>
      <h1 class="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
        ML Research Dashboard
      </h1>
      <p class="text-slate-400 text-sm mt-1">{question}</p>
    </div>
    <div class="text-right">
      <p class="text-slate-500 text-xs">Generated: {timestamp}</p>
      <p class="text-slate-500 text-xs">Session: {session_id}</p>
    </div>
  </div>
</header>

<!-- Stats Bar -->
<div class="max-w-7xl mx-auto px-6 py-6">
  <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
    {stats_cards}
  </div>
</div>

<!-- Search Strategy Summary -->
<div class="max-w-7xl mx-auto px-6 pb-4">
  <div class="card p-4">
    <h3 class="font-semibold text-blue-400 mb-2">Search Strategy</h3>
    <div class="flex flex-wrap gap-2">
      {strategy_badges}
    </div>
    <p class="text-slate-400 text-sm mt-2">Query: <code class="text-green-400">{refined_query}</code></p>
  </div>
</div>

<!-- Main Content: Paper List -->
<div class="max-w-7xl mx-auto px-6 pb-24">
  <div class="flex items-center justify-between mb-4">
    <h2 class="text-xl font-bold">Collected Papers ({total_papers})</h2>
    <div class="flex gap-2">
      <button onclick="selectAll()" class="text-sm px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded-lg">Select All</button>
      <button onclick="deselectAll()" class="text-sm px-3 py-1 bg-slate-600 hover:bg-slate-700 rounded-lg">Deselect</button>
      <button onclick="exportSelection()" class="text-sm px-3 py-1 bg-green-600 hover:bg-green-700 rounded-lg">Export Selected</button>
    </div>
  </div>
  <div class="grid gap-3" id="paper-list">
    {paper_cards}
  </div>
</div>

<!-- Selection Bar -->
<div id="selection-bar">
  <div class="max-w-7xl mx-auto flex items-center justify-between">
    <span id="selection-count" class="text-slate-300">0 papers selected</span>
    <div class="flex gap-3">
      <button onclick="copySelection()" class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg font-semibold text-sm">
        Copy IDs
      </button>
      <button onclick="saveSelection()" class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-semibold text-sm">
        Save to File
      </button>
    </div>
  </div>
</div>

<script>
const papers = {papers_json};
const selected = new Set();

function toggleSelect(arxivId) {{
  const card = document.getElementById('card-' + CSS.escape(arxivId));
  if (selected.has(arxivId)) {{
    selected.delete(arxivId);
    card.classList.remove('selected');
  }} else {{
    selected.add(arxivId);
    card.classList.add('selected');
  }}
  updateSelectionBar();
}}

function selectAll() {{
  papers.forEach(p => {{ if (p.arxiv_id) selected.add(p.arxiv_id); }});
  document.querySelectorAll('.paper-card').forEach(c => c.classList.add('selected'));
  updateSelectionBar();
}}

function deselectAll() {{
  selected.clear();
  document.querySelectorAll('.paper-card').forEach(c => c.classList.remove('selected'));
  updateSelectionBar();
}}

function updateSelectionBar() {{
  const bar = document.getElementById('selection-bar');
  const count = document.getElementById('selection-count');
  if (selected.size > 0) {{
    bar.style.display = 'block';
    count.textContent = selected.size + ' papers selected';
  }} else {{
    bar.style.display = 'none';
  }}
}}

function copySelection() {{
  navigator.clipboard.writeText(JSON.stringify([...selected], null, 2));
  alert('Copied ' + selected.size + ' paper IDs to clipboard!');
}}

function saveSelection() {{
  const data = {{ selected_papers: [...selected], timestamp: new Date().toISOString() }};
  const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'selected_papers.json'; a.click();
}}

function exportSelection() {{ saveSelection(); }}
</script>

</body>
</html>"""


class DashboardGenerator:
    """Generates interactive HTML research dashboards.

    Usage:
        gen = DashboardGenerator()
        path = gen.generate(
            question="RAG hallucination",
            session_id="abc123",
            strategy={...},
            papers=[...],
            kg_stats={...},
            wiki_stats={...},
            sources=["arxiv", "semantic_scholar"],
        )
        print(f"Dashboard: {path}")
    """

    def generate(
        self,
        question: str,
        session_id: str,
        strategy: dict,
        papers: list[dict],
        kg_stats: dict | None = None,
        wiki_stats: dict | None = None,
        sources: list[str] | None = None,
        output_dir: Path | None = None,
    ) -> Path:
        """Generate the dashboard HTML file.

        Args:
            question: Original research question.
            session_id: Session identifier.
            strategy: Search strategy dict.
            papers: List of paper dicts (must have arxiv_id, title, abstract, etc).
            kg_stats: Knowledge graph statistics.
            wiki_stats: Wiki statistics.
            sources: Sources that were searched.
            output_dir: Output directory (default: data/dashboards/).

        Returns:
            Path to the generated HTML file.
        """
        output_dir = output_dir or config.DATA_DIR / "dashboards"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        refined_query = strategy.get("refined_query", question)

        # Stats cards
        stats_cards = self._build_stats_cards(
            total_papers=len(papers),
            sources=sources or [],
            kg_stats=kg_stats,
            wiki_stats=wiki_stats,
        )

        # Strategy badges
        strategy_badges = self._build_strategy_badges(strategy)

        # Paper cards
        paper_cards = self._build_paper_cards(papers)

        html = DASHBOARD_TEMPLATE.format(
            title=question[:60],
            question=question,
            timestamp=timestamp,
            session_id=session_id,
            stats_cards=stats_cards,
            strategy_badges=strategy_badges,
            refined_query=refined_query,
            total_papers=len(papers),
            paper_cards=paper_cards,
            papers_json=json.dumps([
                {"arxiv_id": p.get("arxiv_id", p.get("paper_id", "")),
                 "title": p.get("title", "")}
                for p in papers
            ], ensure_ascii=False),
        )

        filename = f"dashboard_{session_id}_{int(time.time())}.html"
        output_path = output_dir / filename
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def _build_stats_cards(
        self,
        total_papers: int,
        sources: list[str],
        kg_stats: dict | None,
        wiki_stats: dict | None,
    ) -> str:
        cards = [
            f'<div class="card p-4 text-center"><p class="stat-number">{total_papers}</p><p class="text-slate-400 text-sm">Papers Found</p></div>',
            f'<div class="card p-4 text-center"><p class="stat-number">{len(sources)}</p><p class="text-slate-400 text-sm">Sources</p></div>',
        ]
        if kg_stats:
            cards.append(
                f'<div class="card p-4 text-center"><p class="stat-number">{kg_stats.get("nodes", 0)}</p><p class="text-slate-400 text-sm">KG Nodes</p></div>'
            )
            cards.append(
                f'<div class="card p-4 text-center"><p class="stat-number">{kg_stats.get("edges", 0)}</p><p class="text-slate-400 text-sm">KG Edges</p></div>'
            )
        if wiki_stats:
            cards.append(
                f'<div class="card p-4 text-center"><p class="stat-number">{wiki_stats.get("indexed", 0)}</p><p class="text-slate-400 text-sm">Wiki Articles</p></div>'
            )
        return "\n".join(cards)

    def _build_strategy_badges(self, strategy: dict) -> str:
        badges = []
        for kw in strategy.get("keywords", [])[:5]:
            badges.append(f'<span class="badge bg-blue-900 text-blue-300">{kw}</span>')
        for src in strategy.get("sources", []):
            badges.append(f'<span class="badge source-{src}">{src}</span>')
        for domain in strategy.get("domains", []):
            badges.append(f'<span class="badge bg-purple-900 text-purple-300">{domain}</span>')
        return "\n".join(badges)

    def _build_paper_cards(self, papers: list[dict]) -> str:
        cards = []
        for i, p in enumerate(papers):
            arxiv_id = p.get("arxiv_id", p.get("paper_id", f"unknown_{i}"))
            title = p.get("title", "Untitled")
            abstract = (p.get("abstract") or "")[:200] + "..." if len(p.get("abstract") or "") > 200 else (p.get("abstract") or "")
            year = p.get("year", "")
            source = p.get("source", "unknown")
            categories = p.get("categories", [])
            authors = p.get("authors", [])

            cat_html = " ".join(
                f'<span class="badge bg-slate-700 text-slate-300">{c}</span>'
                for c in (categories or [])[:3]
            )
            authors_str = ", ".join((authors or [])[:3])
            if len(authors or []) > 3:
                authors_str += f" +{len(authors) - 3}"

            cards.append(f'''
            <div class="card paper-card p-4 fade-in" id="card-{arxiv_id}"
                 onclick="toggleSelect('{arxiv_id}')">
              <div class="flex items-start gap-3">
                <input type="checkbox" class="mt-1 w-4 h-4 accent-green-500"
                       onclick="event.stopPropagation(); toggleSelect('{arxiv_id}')"
                       id="cb-{arxiv_id}">
                <div class="flex-1 min-w-0">
                  <div class="flex items-center gap-2 mb-1">
                    <span class="badge source-{source}">{source}</span>
                    {f'<span class="text-slate-500 text-xs">{year}</span>' if year else ''}
                    <span class="text-slate-600 text-xs font-mono">{arxiv_id}</span>
                  </div>
                  <h3 class="font-semibold text-white text-sm leading-tight mb-1">{title}</h3>
                  <p class="text-slate-400 text-xs leading-relaxed mb-2">{abstract}</p>
                  <div class="flex items-center gap-2 flex-wrap">
                    {cat_html}
                    {f'<span class="text-slate-500 text-xs">{authors_str}</span>' if authors_str else ''}
                  </div>
                </div>
              </div>
            </div>''')
        return "\n".join(cards)
