"""ML Research Platform — Research Map module.

Generates interactive visual maps of research landscapes from
knowledge graphs, showing topic clusters, paper relationships,
and research frontiers. Each node is clickable to reveal full details
in a side panel.
"""

from __future__ import annotations

import json
import os
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path

from pyvis.network import Network

from ml_platform.config import AppConfig


# ── Data models ──────────────────────────────────────────────────────


class ClusterInfo:
    """A cluster of related research topics."""
    __slots__ = ("id", "label", "size", "color", "papers", "keywords")

    def __init__(self, cluster_id: int, label: str):
        self.id = cluster_id
        self.label = label
        self.size = 0
        self.color = ""
        self.papers: list[str] = []
        self.keywords: list[str] = []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "size": self.size,
            "papers": self.papers,
            "keywords": self.keywords,
        }


class ResearchMapResult:
    """Result of a research map generation."""
    __slots__ = ("topic", "total_papers", "clusters", "edges", "html_path")

    def __init__(self, topic: str):
        self.topic = topic
        self.total_papers = 0
        self.clusters: list[ClusterInfo] = []
        self.edges: int = 0
        self.html_path: str = ""


# ── Color palettes ───────────────────────────────────────────────────

CLUSTER_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
]

NODE_COLORS = {
    "Paper": "#E74C3C", "Author": "#3498DB", "Method": "#2ECC71",
    "Dataset": "#F39C12", "Concept": "#9B59B6", "Institution": "#1ABC9C",
    "Venue": "#E67E22",
}

NODE_SHAPES = {
    "Paper": "dot", "Author": "diamond", "Method": "triangle",
    "Concept": "triangleDown", "Dataset": "square", "Institution": "hexagon",
    "Venue": "star",
}


# ── Community detection (label propagation) ─────────────────────────


def _detect_communities(
    adjacency: dict[str, set[str]],
    max_iterations: int = 20,
) -> dict[str, int]:
    """Label-propagation community detection."""
    nodes = list(adjacency.keys())
    if not nodes:
        return {}
    community = {n: i for i, n in enumerate(nodes)}
    for _ in range(max_iterations):
        changed = False
        for node in nodes:
            neighbors = adjacency.get(node, set())
            if not neighbors:
                continue
            neighbor_comms = Counter(community.get(nb, 0) for nb in neighbors)
            best_comm = neighbor_comms.most_common(1)[0][0]
            if community[node] != best_comm:
                community[node] = best_comm
                changed = True
        if not changed:
            break
    unique_comms = sorted(set(community.values()))
    comm_map = {c: i for i, c in enumerate(unique_comms)}
    return {n: comm_map[community[n]] for n in nodes}


# ── Detail panel HTML/JS ────────────────────────────────────────────

DETAIL_PANEL_CSS = """
<style>
  #detail-panel {
    position: fixed;
    top: 0;
    right: -520px;
    width: 500px;
    height: 100vh;
    background: #16213e;
    border-left: 2px solid #4ECDC4;
    padding: 20px;
    overflow-y: auto;
    z-index: 1000;
    transition: right 0.3s ease;
    font-family: 'Courier New', monospace;
    color: #e0e0e0;
  }
  #detail-panel.open { right: 0; }
  #detail-panel .close-btn {
    position: absolute; top: 10px; right: 15px;
    background: #e74c3c; color: white; border: none;
    border-radius: 50%; width: 28px; height: 28px;
    cursor: pointer; font-size: 14px; font-weight: bold;
  }
  #detail-panel .close-btn:hover { background: #c0392b; }
  #detail-panel h3 {
    color: #4ECDC4; margin: 0 0 12px 0;
    padding-bottom: 8px; border-bottom: 1px solid #333;
    padding-right: 35px;
  }
  #detail-panel .meta { font-size: 12px; color: #888; margin-bottom: 10px; }
  #detail-panel .section {
    margin: 12px 0; padding: 10px;
    background: #1a1a2e; border-radius: 6px;
  }
  #detail-panel .section-title {
    color: #FFEAA7; font-weight: bold;
    font-size: 13px; margin-bottom: 6px;
  }
  #detail-panel .content { font-size: 12px; line-height: 1.7; }
  #detail-panel .tag {
    display: inline-block; background: #2a2a4e; color: #4ECDC4;
    padding: 2px 8px; border-radius: 10px; margin: 2px;
    font-size: 11px;
  }
  #detail-panel .badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 11px; font-weight: bold; margin-right: 5px;
  }
  #detail-panel .badge-paper { background: #E74C3C; color: white; }
  #detail-panel .badge-author { background: #3498DB; color: white; }
  #detail-panel .badge-method { background: #2ECC71; color: white; }
  #detail-panel .badge-concept { background: #9B59B6; color: white; }
  #detail-panel .badge-institution { background: #1ABC9C; color: white; }
  #detail-panel .badge-venue { background: #E67E22; color: white; }
  #detail-panel .badge-category { background: #34495E; color: #e0e0e0; }
  #detail-panel a { color: #45B7D1; text-decoration: none; }
  #detail-panel a:hover { text-decoration: underline; }
</style>
"""

DETAIL_PANEL_HTML = """
<div id="detail-panel">
  <button class="close-btn" onclick="closeDetail()">&times;</button>
  <div id="detail-content">
    <p style="color:#888;">Click a node to see details</p>
  </div>
</div>
"""

DETAIL_PANEL_JS = """
<script>
// Node detail data injected from Python
const NODE_DETAILS = __NODE_DATA__;

function showDetail(nodeId) {
  const data = NODE_DETAILS[nodeId];
  if (!data) return;
  const panel = document.getElementById('detail-panel');
  const content = document.getElementById('detail-content');
  let html = '';

  // Badge + Title
  const badgeClass = 'badge-' + (data.type || 'paper').toLowerCase();
  html += '<h3><span class="badge ' + badgeClass + '">' + (data.type || 'Node') + '</span> ' + (data.title || nodeId) + '</h3>';
  html += '<div class="meta">' + (data.meta || '') + '</div>';

  // Abstract / Description
  if (data.abstract) {
    html += '<div class="section"><div class="section-title">Abstract</div><div class="content">' + data.abstract + '</div></div>';
  }
  if (data.description) {
    html += '<div class="section"><div class="section-title">Description</div><div class="content">' + data.description + '</div></div>';
  }

  // Authors
  if (data.authors && data.authors.length > 0) {
    html += '<div class="section"><div class="section-title">Authors</div><div class="content">' + data.authors.join(', ') + '</div></div>';
  }

  // Categories / Keywords
  if (data.categories && data.categories.length > 0) {
    html += '<div class="section"><div class="section-title">Categories</div><div class="content">';
    data.categories.forEach(function(c) { html += '<span class="tag">' + c + '</span>'; });
    html += '</div></div>';
  }
  if (data.keywords && data.keywords.length > 0) {
    html += '<div class="section"><div class="section-title">Keywords</div><div class="content">';
    data.keywords.forEach(function(k) { html += '<span class="tag">' + k + '</span>'; });
    html += '</div></div>';
  }

  // Related papers
  if (data.related && data.related.length > 0) {
    html += '<div class="section"><div class="section-title">Related (' + data.related.length + ')</div><div class="content" style="max-height:200px;overflow-y:auto;">';
    data.related.forEach(function(r) { html += '<div style="margin:3px 0;padding:3px 6px;background:#2a2a4e;border-radius:3px;">' + r + '</div>'; });
    html += '</div></div>';
  }

  // Links
  if (data.url) {
    html += '<div class="section"><div class="section-title">Links</div><div class="content"><a href="' + data.url + '" target="_blank">' + data.url + '</a></div></div>';
  }

  // Raw properties
  if (data.extra && Object.keys(data.extra).length > 0) {
    html += '<div class="section"><div class="section-title">Properties</div><div class="content"><table style="width:100%;font-size:11px;">';
    for (const [k, v] of Object.entries(data.extra)) {
      if (v && v !== 'None' && v !== 'null') {
        html += '<tr><td style="color:#888;padding:2px 8px 2px 0;">' + k + '</td><td>' + v + '</td></tr>';
      }
    }
    html += '</table></div></div>';
  }

  content.innerHTML = html;
  panel.classList.add('open');
}

function closeDetail() {
  document.getElementById('detail-panel').classList.remove('open');
}

// Bind click events to pyvis network
function bindClickEvents() {
  // pyvis stores network in window.network
  if (typeof network !== 'undefined') {
    network.on('click', function(params) {
      if (params.nodes && params.nodes.length > 0) {
        showDetail(params.nodes[0]);
      }
    });
  } else {
    setTimeout(bindClickEvents, 500);
  }
}
setTimeout(bindClickEvents, 1000);
</script>
"""


# ── Research Map Builder ────────────────────────────────────────────


class ResearchMapBuilder:
    """Builds interactive research landscape maps.

    Each node is clickable to show full details in a side panel.
    """

    def __init__(self, data_dir: Path | str | None = None):
        if data_dir is None:
            data_dir = AppConfig.DATA_DIR / "maps"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _inject_detail_panel(self, html_path: Path, node_details: dict) -> None:
        """Inject the detail panel CSS/HTML/JS into the saved HTML."""
        html = html_path.read_text()

        # Serialize node details as JSON
        details_json = json.dumps(node_details, ensure_ascii=False, default=str)
        js_with_data = DETAIL_PANEL_JS.replace("__NODE_DATA__", details_json)

        # Inject before </head> and </body>
        html = html.replace("</head>", DETAIL_PANEL_CSS + "\n</head>")
        html = html.replace("</body>", DETAIL_PANEL_HTML + "\n" + js_with_data + "\n</body>")
        html_path.write_text(html)

    # ── KG-based map ────────────────────────────────────────────────

    def build_from_graph(
        self,
        topic: str,
        open_browser: bool = True,
        height: str = "900px",
    ) -> ResearchMapResult:
        """Build research map from a knowledge graph."""
        from ml_platform.graph.knowledge_graph import KnowledgeGraph

        result = ResearchMapResult(topic)
        kg = KnowledgeGraph.open(topic)
        stats = kg.get_stats()

        if stats.node_count == 0:
            print(f"  [WARN] Knowledge graph '{topic}' is empty")
            return result

        # Fetch nodes and edges
        nodes_data = kg.query("MATCH (n) RETURN n.node_id, n.label, labels(n) as types")
        edges_data = kg.query("MATCH (a)-[r]->(b) RETURN a.node_id, b.node_id, type(r) as rel_type")
        kg.close()

        # Build lookup
        node_map: dict[str, dict] = {}
        node_details: dict[str, dict] = {}

        for row in nodes_data:
            nid = row.get("n.node_id", "")
            label = row.get("n.label", nid)
            types_raw = row.get("types", "[]")
            types = json.loads(types_raw) if isinstance(types_raw, str) else types_raw
            node_map[nid] = {"id": nid, "label": label, "types": types}

        # Build adjacency
        adjacency: dict[str, set[str]] = defaultdict(set)
        edge_list = []
        for row in edges_data:
            src = row.get("a.node_id", "")
            tgt = row.get("b.node_id", "")
            rel = row.get("rel_type", "RELATED")
            if src in node_map and tgt in node_map:
                adjacency[src].add(tgt)
                adjacency[tgt].add(src)
                edge_list.append((src, tgt, rel))

        # Community detection
        communities = _detect_communities(adjacency)

        # Build node details for detail panel
        # Gather relations per node
        node_relations: dict[str, list[str]] = defaultdict(list)
        for src, tgt, rel in edge_list:
            src_label = node_map[src]["label"] if src in node_map else src
            tgt_label = node_map[tgt]["label"] if tgt in node_map else tgt
            node_relations[src].append(f"[{rel}] {tgt_label}")
            node_relations[tgt].append(f"[{rel}←] {src_label}")

        for nid, node in node_map.items():
            types = node["types"]
            main_type = types[0] if types else "Unknown"
            related = node_relations.get(nid, [])[:10]

            node_details[nid] = {
                "type": main_type,
                "title": node["label"][:80],
                "meta": f"Type: {main_type} | ID: {nid}",
                "description": node["label"],
                "related": related,
                "extra": {
                    "Node ID": nid,
                    "Types": ", ".join(types),
                    "Cluster": str(communities.get(nid, 0)),
                    "Connections": str(len(adjacency.get(nid, set()))),
                },
            }

        # Cluster info
        comm_nodes: dict[int, list[str]] = defaultdict(list)
        for nid, cid in communities.items():
            comm_nodes[cid].append(nid)

        clusters: list[ClusterInfo] = []
        for cid in sorted(comm_nodes.keys()):
            nids = comm_nodes[cid]
            type_counts: Counter = Counter()
            keywords: list[str] = []
            paper_ids: list[str] = []
            for nid in nids:
                node = node_map[nid]
                for t in node["types"]:
                    type_counts[t] += 1
                keywords.append(node["label"][:40])
                if "Paper" in node["types"]:
                    paper_ids.append(nid)
            dominant = type_counts.most_common(1)[0][0] if type_counts else "Unknown"
            cluster = ClusterInfo(cid, f"{dominant} #{cid}")
            cluster.size = len(nids)
            cluster.color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            cluster.papers = paper_ids
            cluster.keywords = [k for k, _ in Counter(keywords).most_common(5)]
            clusters.append(cluster)

        result.clusters = clusters
        result.total_papers = stats.papers_indexed
        result.edges = len(edge_list)

        # ── Build pyvis network ──
        net = Network(
            height=height, width="100%", directed=False,
            bgcolor="#1a1a2e", font_color="#e0e0e0",
        )
        net.heading = f"Research Map: {topic}"

        for nid, node in node_map.items():
            comm_id = communities.get(nid, 0)
            types = node["types"]
            main_type = types[0] if types else "Unknown"
            cluster_color = CLUSTER_COLORS[comm_id % len(CLUSTER_COLORS)]
            shape = NODE_SHAPES.get(main_type, "dot")
            color = NODE_COLORS.get(main_type, cluster_color)
            size = {"Paper": 22, "Author": 14, "Concept": 16, "Method": 16}.get(main_type, 10)

            # Hover tooltip (brief)
            tooltip = f"<b>{node['label'][:60]}</b><br><i>{main_type}</i> — Click for details"

            net.add_node(nid, label=node["label"][:25], title=tooltip,
                         shape=shape, size=size, color=color, group=comm_id)

        for src, tgt, rel in edge_list:
            net.add_edge(src, tgt, title=rel, color="#334455", width=1)

        net.force_atlas_2based(gravity=-80, central_gravity=0.01,
                               spring_length=150, spring_strength=0.05, damping=0.4)

        # Save and inject detail panel
        output_path = self.data_dir / f"map_{topic}.html"
        net.save_graph(str(output_path))

        # Add legend header
        legend_items = []
        for c in clusters:
            kw = ", ".join(c.keywords[:3])
            legend_items.append(
                f'<span style="color:{c.color};">&#9679;</span> '
                f"<b>{c.label}</b> ({c.size}) — {kw}"
            )

        header_html = f"""
        <div style="background:#16213e;padding:15px;margin:10px;border-radius:8px;color:#e0e0e0;font-family:monospace;">
          <h2 style="margin:0 0 10px 0;color:#4ECDC4;">Research Map: {topic}</h2>
          <div style="font-size:12px;">
            <b>Papers:</b> {result.total_papers} | <b>Nodes:</b> {stats.node_count} |
            <b>Edges:</b> {result.edges} | <b>Clusters:</b> {len(clusters)}
          </div>
          <hr style="border-color:#333;">
          <div style="font-size:11px;line-height:1.6;">{"<br>".join(legend_items)}</div>
          <hr style="border-color:#333;">
          <div style="font-size:10px;color:#888;">
            Click any node to see full details | Scroll to zoom, drag to move
          </div>
        </div>
        """
        html = output_path.read_text()
        html = html.replace("<body>", f"<body>\n{header_html}")
        output_path.write_text(html)

        self._inject_detail_panel(output_path, node_details)
        result.html_path = str(output_path)

        if open_browser:
            webbrowser.open(f"file://{output_path.resolve()}")

        return result

    # ── Paper-based map ─────────────────────────────────────────────

    def build_from_papers(
        self,
        topic: str,
        papers: list | None = None,
        open_browser: bool = True,
    ) -> ResearchMapResult:
        """Build research map from papers in DB.

        Two-layer network:
          - Layer 1: Category nodes (co-occurrence network)
          - Layer 2: Paper nodes connected to their categories
        Each paper node has full abstract, authors, etc.
        """
        from ml_platform.db import PapersDB

        result = ResearchMapResult(topic)
        db = PapersDB()
        all_papers = papers or db.get_papers(limit=1000)
        result.total_papers = len(all_papers)

        if not all_papers:
            return result

        # Index papers by category
        cat_papers: dict[str, list] = defaultdict(list)
        paper_cats: dict[str, list[str]] = {}

        for paper in all_papers:
            pid = paper.paper_id
            cats = paper.categories if paper.categories else ["uncategorized"]
            paper_cats[pid] = cats
            for cat in cats:
                cat_papers[cat].append(paper)

        # Build co-category edges
        edge_set: set[tuple[str, str]] = set()
        cat_adjacency: dict[str, set[str]] = defaultdict(set)

        for pid, cats in paper_cats.items():
            for i, c1 in enumerate(cats):
                for c2 in cats[i + 1:]:
                    pair = tuple(sorted([c1, c2]))
                    if pair not in edge_set:
                        edge_set.add(pair)
                        cat_adjacency[c1].add(c2)
                        cat_adjacency[c2].add(c1)

        # Community detection
        all_cats = list(cat_papers.keys())
        cat_communities = _detect_communities(cat_adjacency)

        # ── Build pyvis network ──
        net = Network(
            height="900px", width="100%", directed=False,
            bgcolor="#1a1a2e", font_color="#e0e0e0",
        )
        net.heading = f"Research Landscape: {topic}"

        node_details: dict[str, dict] = {}

        # Layer 1: Category nodes
        for cat in all_cats:
            comm_id = cat_communities.get(cat, 0)
            color = CLUSTER_COLORS[comm_id % len(CLUSTER_COLORS)]
            pcount = len(cat_papers[cat])
            years = sorted(set(
                p.year for p in cat_papers[cat] if p.year
            ))
            year_str = f"{years[0]}-{years[-1]}" if years else "N/A"

            tooltip = f"<b>{cat}</b><br>{pcount} papers ({year_str})<br><i>Click for paper list</i>"
            net.add_node(cat, label=cat, title=tooltip,
                         shape="dot", size=min(15 + pcount * 3, 50),
                         color=color, group=comm_id)

            paper_titles = [p.title[:60] for p in cat_papers[cat][:15]]
            node_details[cat] = {
                "type": "Category",
                "title": cat,
                "meta": f"{pcount} papers | {year_str}",
                "description": f"Research category: {cat}",
                "related": paper_titles,
                "extra": {
                    "Papers": str(pcount),
                    "Period": year_str,
                    "Cluster": str(comm_id),
                },
            }

        # Layer 2: Paper nodes (smaller, connected to categories)
        for paper in all_papers:
            pid = paper.paper_id
            cats = paper_cats.get(pid, [])
            if not cats:
                continue

            authors_list = [a.name if hasattr(a, "name") else str(a)
                            for a in (paper.authors or [])]

            abstract = (paper.abstract or "")[:300]
            title = paper.title or pid

            # First category determines color
            first_cat = cats[0]
            comm_id = cat_communities.get(first_cat, 0)
            color = CLUSTER_COLORS[comm_id % len(CLUSTER_COLORS)]

            tooltip = f"<b>{title[:50]}</b><br>{', '.join(cats)}<br><i>Click for details</i>"
            net.add_node(pid, label=title[:20], title=tooltip,
                         shape="diamond", size=8, color="#556677", group=comm_id)

            # Paper → Category edges
            for cat in cats:
                net.add_edge(pid, cat, color="#223344", width=0.5)

            node_details[pid] = {
                "type": "Paper",
                "title": title,
                "meta": f"{paper.year or 'N/A'} | {', '.join(cats)} | Cited: {paper.citation_count or 0}",
                "abstract": abstract,
                "authors": authors_list[:10],
                "categories": cats,
                "url": paper.url or "",
                "extra": {
                    "Year": str(paper.year or ""),
                    "Source": paper.source or "",
                    "ArXiv": paper.arxiv_id or "",
                    "Citations": str(paper.citation_count or 0),
                    "Score": f"{paper.composite_score:.2f}" if paper.composite_score else "",
                },
            }

        # Category co-occurrence edges
        for c1, c2 in edge_set:
            shared = len(set(p.paper_id for p in cat_papers[c1]) &
                         set(p.paper_id for p in cat_papers[c2]))
            net.add_edge(c1, c2, title=f"shared: {shared}", width=min(shared, 5), color="#334455")

        result.edges = len(edge_set)

        # Cluster info
        comm_cats: dict[int, list[str]] = defaultdict(list)
        for cat, cid in cat_communities.items():
            comm_cats[cid].append(cat)
        for cid in sorted(comm_cats.keys()):
            cats_in = comm_cats[cid]
            cluster = ClusterInfo(cid, f"Cluster {cid}: {cats_in[0]}")
            cluster.size = len(cats_in)
            cluster.color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            cluster.papers = [str(p.paper_id) for c in cats_in for p in cat_papers[c][:3]]
            cluster.keywords = cats_in[:5]
            result.clusters.append(cluster)

        # Save
        output_path = self.data_dir / f"map_{topic}.html"
        net.save_graph(str(output_path))

        # Header + legend
        legend_items = []
        for c in result.clusters:
            kw = ", ".join(c.keywords)
            legend_items.append(
                f'<span style="color:{c.color};">&#9679;</span> '
                f"<b>{c.label}</b> ({c.size} cats) — {kw}"
            )
        header_html = f"""
        <div style="background:#16213e;padding:15px;margin:10px;border-radius:8px;color:#e0e0e0;font-family:monospace;">
          <h2 style="margin:0 0 10px 0;color:#4ECDC4;">Research Landscape: {topic}</h2>
          <div style="font-size:12px;">
            <b>Papers:</b> {result.total_papers} | <b>Categories:</b> {len(all_cats)} |
            <b>Edges:</b> {result.edges} | <b>Clusters:</b> {len(result.clusters)}
          </div>
          <hr style="border-color:#333;">
          <div style="font-size:11px;line-height:1.6;">{"<br>".join(legend_items)}</div>
          <hr style="border-color:#333;">
          <div style="font-size:10px;color:#888;">
            ● Category nodes (large) | ◆ Paper nodes (small) | Click any node for full details
          </div>
        </div>
        """
        html = output_path.read_text()
        html = html.replace("<body>", f"<body>\n{header_html}")
        output_path.write_text(html)

        self._inject_detail_panel(output_path, node_details)
        result.html_path = str(output_path)

        if open_browser:
            webbrowser.open(f"file://{output_path.resolve()}")

        return result


def generate_map_markdown(result: ResearchMapResult) -> str:
    """Generate markdown summary of a research map."""
    lines = [
        f"# Research Map: {result.topic}",
        "",
        f"- **Papers:** {result.total_papers}",
        f"- **Edges:** {result.edges}",
        f"- **Clusters:** {len(result.clusters)}",
        "",
        "## Clusters",
        "",
    ]
    for cluster in sorted(result.clusters, key=lambda c: c.size, reverse=True):
        kw = ", ".join(cluster.keywords[:5])
        lines.append(f"### {cluster.label} ({cluster.size} nodes)")
        lines.append(f"- Keywords: {kw}")
        lines.append(f"- Papers: {len(cluster.papers)}")
        lines.append("")
    if result.html_path:
        lines.append(f"**Interactive map:** `{result.html_path}`")
    return "\n".join(lines)
