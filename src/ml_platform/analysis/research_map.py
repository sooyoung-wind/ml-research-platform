"""ML Research Platform — Research Map module.

Generates interactive visual maps of research landscapes from
knowledge graphs, showing topic clusters, paper relationships,
and research frontiers. Each node is clickable to reveal full details
in a side panel.
"""

from __future__ import annotations

import json
import os
import re
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


# ── Detail panel HTML/JS (Tailwind CSS) ────────────────────────────

TAILWIND_CDN = """<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
  theme: {
    extend: {
      colors: {
        dark: { 900: '#0f0f1a', 800: '#161625', 700: '#1e1e35', 600: '#2a2a4e' },
        accent: { cyan: '#4ECDC4', yellow: '#FFEAA7', blue: '#45B7D1', red: '#FF6B6B' },
      },
      fontFamily: { mono: ['JetBrains Mono', 'Fira Code', 'monospace'] },
    },
  },
};
</script>
"""

DETAIL_PANEL_CSS = """
<style>
  /* ── Detail panel slide-in ── */
  #detail-panel {
    position: fixed; top: 0; right: -540px; width: 520px; height: 100vh;
    transition: right 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 1000;
  }
  #detail-panel.open { right: 0; }

  /* ── Overlay for mobile feel ── */
  #detail-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.4);
    z-index: 999; opacity: 0; pointer-events: none;
    transition: opacity 0.3s ease;
  }
  #detail-overlay.open { opacity: 1; pointer-events: auto; }

  /* ── Scrollbar styling ── */
  #detail-panel::-webkit-scrollbar { width: 6px; }
  #detail-panel::-webkit-scrollbar-track { background: #161625; }
  #detail-panel::-webkit-scrollbar-thumb { background: #4ECDC4; border-radius: 3px; }

  /* ── pyvis canvas padding for panel ── */
  body { margin: 0; overflow: hidden; }
</style>
"""

DETAIL_PANEL_HTML = """
<!-- Dark overlay -->
<div id="detail-overlay" onclick="closeDetail()"></div>

<!-- Detail side panel -->
<div id="detail-panel" class="bg-dark-800 border-l-2 border-accent-cyan font-mono text-gray-200 overflow-y-auto">
  <div id="detail-content" class="p-6">
    <div class="flex flex-col items-center justify-center h-48 text-gray-500">
      <svg class="w-10 h-10 mb-3 opacity-30" fill="currentColor" viewBox="0 0 20 20">
        <path d="M10 2a8 8 0 100 16 8 8 0 000-16zm1 11H9v-2h2v2zm0-4H9V5h2v4z"/>
      </svg>
      <p class="text-sm">Click a node to see details</p>
    </div>
  </div>
</div>
"""

DETAIL_PANEL_JS = """
<script>
const NODE_DETAILS = __NODE_DATA__;

// Icon SVGs for each type
const TYPE_ICONS = {
  Paper: '<svg class="w-4 h-4 inline mr-1" fill="currentColor" viewBox="0 0 20 20"><path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/><path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clip-rule="evenodd"/></svg>',
  Category: '<svg class="w-4 h-4 inline mr-1" fill="currentColor" viewBox="0 0 20 20"><path d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z"/></svg>',
  Author: '<svg class="w-4 h-4 inline mr-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd"/></svg>',
  Concept: '<svg class="w-4 h-4 inline mr-1" fill="currentColor" viewBox="0 0 20 20"><path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM3 10a1 1 0 011-1h1a1 1 0 110 2H4a1 1 0 01-1-1z"/><path fill-rule="evenodd" d="M10 6a4 4 0 100 8 4 4 0 000-8z" clip-rule="evenodd"/></svg>',
  Institution: '<svg class="w-4 h-4 inline mr-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4 4a2 2 0 012-2h8a2 2 0 012 2v12l-6-3-6 3V4z" clip-rule="evenodd"/></svg>',
  Method: '<svg class="w-4 h-4 inline mr-1" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clip-rule="evenodd"/></svg>',
  Venue: '<svg class="w-4 h-4 inline mr-1" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/></svg>',
};

const BADGE_COLORS = {
  Paper:       'bg-red-500/80 text-white',
  Category:    'bg-gray-600 text-gray-200',
  Author:      'bg-blue-500/80 text-white',
  Concept:     'bg-purple-500/80 text-white',
  Institution: 'bg-teal-500/80 text-white',
  Method:      'bg-green-500/80 text-white',
  Venue:       'bg-orange-500/80 text-white',
};

function esc(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function showDetail(nodeId) {
  const data = NODE_DETAILS[nodeId];
  if (!data) return;
  const panel = document.getElementById('detail-panel');
  const content = document.getElementById('detail-content');
  let h = '';

  // ── Close button ──
  h += '<button onclick="closeDetail()" class="absolute top-3 right-3 w-8 h-8 rounded-full bg-red-500/80 hover:bg-red-600 text-white flex items-center justify-center text-sm font-bold transition-colors">&times;</button>';

  // ── Header: badge + title ──
  const icon = TYPE_ICONS[data.type] || '';
  const badge = BADGE_COLORS[data.type] || 'bg-gray-600 text-gray-200';
  h += '<div class="mb-4 pr-10">';
  h += '<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold ' + badge + '">';
  h += icon + ' ' + esc(data.type || 'Node') + '</span>';
  h += '<h3 class="mt-2 text-lg font-semibold text-accent-cyan leading-snug">' + esc(data.title || nodeId) + '</h3>';
  h += '<p class="mt-1 text-xs text-gray-500">' + esc(data.meta || '') + '</p>';
  h += '</div>';

  // ── Divider ──
  h += '<hr class="border-dark-600 mb-4">';

  // ── Abstract ──
  if (data.abstract) {
    h += '<div class="mb-4 bg-dark-700 rounded-lg p-3 border border-dark-600">';
    h += '<h4 class="text-xs font-bold text-accent-yellow uppercase tracking-wider mb-2">Abstract</h4>';
    h += '<p class="text-sm text-gray-300 leading-relaxed">' + esc(data.abstract) + '</p>';
    h += '</div>';
  }

  // ── Description ──
  if (data.description && !data.abstract) {
    h += '<div class="mb-4 bg-dark-700 rounded-lg p-3 border border-dark-600">';
    h += '<h4 class="text-xs font-bold text-accent-yellow uppercase tracking-wider mb-2">Description</h4>';
    h += '<p class="text-sm text-gray-300 leading-relaxed">' + esc(data.description) + '</p>';
    h += '</div>';
  }

  // ── Authors ──
  if (data.authors && data.authors.length > 0) {
    h += '<div class="mb-4">';
    h += '<h4 class="text-xs font-bold text-accent-yellow uppercase tracking-wider mb-2">Authors</h4>';
    h += '<div class="flex flex-wrap gap-1">';
    data.authors.forEach(function(a) {
      h += '<span class="inline-flex items-center bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded-full text-xs">' + esc(a) + '</span>';
    });
    h += '</div></div>';
  }

  // ── Categories ──
  if (data.categories && data.categories.length > 0) {
    h += '<div class="mb-4">';
    h += '<h4 class="text-xs font-bold text-accent-yellow uppercase tracking-wider mb-2">Categories</h4>';
    h += '<div class="flex flex-wrap gap-1">';
    data.categories.forEach(function(c) {
      h += '<span class="inline-flex items-center bg-accent-cyan/20 text-accent-cyan px-2.5 py-0.5 rounded-full text-xs font-medium">' + esc(c) + '</span>';
    });
    h += '</div></div>';
  }

  // ── Keywords ──
  if (data.keywords && data.keywords.length > 0) {
    h += '<div class="mb-4">';
    h += '<h4 class="text-xs font-bold text-accent-yellow uppercase tracking-wider mb-2">Keywords</h4>';
    h += '<div class="flex flex-wrap gap-1">';
    data.keywords.forEach(function(k) {
      h += '<span class="inline-flex items-center bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded-full text-xs">' + esc(k) + '</span>';
    });
    h += '</div></div>';
  }

  // ── Related papers ──
  if (data.related && data.related.length > 0) {
    h += '<div class="mb-4">';
    h += '<h4 class="text-xs font-bold text-accent-yellow uppercase tracking-wider mb-2">Related <span class="text-gray-500">(' + data.related.length + ')</span></h4>';
    h += '<div class="max-h-48 overflow-y-auto space-y-1 pr-1">';
    data.related.forEach(function(r) {
      h += '<div class="bg-dark-700 border border-dark-600 rounded px-3 py-1.5 text-xs text-gray-300 hover:border-accent-cyan/40 transition-colors">' + esc(r) + '</div>';
    });
    h += '</div></div>';
  }

  // ── URL link ──
  if (data.url) {
    h += '<div class="mb-4">';
    h += '<h4 class="text-xs font-bold text-accent-yellow uppercase tracking-wider mb-2">Link</h4>';
    h += '<a href="' + esc(data.url) + '" target="_blank" class="inline-flex items-center text-accent-blue hover:underline text-sm">';
    h += '<svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>';
    h += esc(data.url) + '</a></div>';
  }

  // ── Properties table ──
  if (data.extra && Object.keys(data.extra).length > 0) {
    h += '<div class="mb-4">';
    h += '<h4 class="text-xs font-bold text-accent-yellow uppercase tracking-wider mb-2">Properties</h4>';
    h += '<table class="w-full text-xs">';
    for (const [k, v] of Object.entries(data.extra)) {
      if (v && v !== 'None' && v !== 'null' && v !== '') {
        h += '<tr class="border-b border-dark-600"><td class="py-1.5 pr-4 text-gray-500 whitespace-nowrap">' + esc(k) + '</td><td class="py-1.5 text-gray-300">' + esc(String(v)) + '</td></tr>';
      }
    }
    h += '</table></div>';
  }

  content.innerHTML = h;
  panel.classList.add('open');
  document.getElementById('detail-overlay').classList.add('open');
}

function closeDetail() {
  document.getElementById('detail-panel').classList.remove('open');
  document.getElementById('detail-overlay').classList.remove('open');
}

// Bind click events
function bindClickEvents() {
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

// ESC key to close
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeDetail();
});
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

        # Remove pyvis default heading (<center><h1>...</h1></center>)
        html = re.sub(r"<center>\s*<h1>[^<]*</h1>\s*</center>", "", html)

        # Serialize node details as JSON
        details_json = json.dumps(node_details, ensure_ascii=False, default=str)
        js_with_data = DETAIL_PANEL_JS.replace("__NODE_DATA__", details_json)

        # Inject: Tailwind CDN + CSS before </head>, HTML + JS before </body>
        html = html.replace("</head>", TAILWIND_CDN + DETAIL_PANEL_CSS + "\n</head>")
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
        net.heading = ""

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

        # Add legend header (Tailwind)
        legend_items_html = ""
        for c in clusters:
            kw = ", ".join(c.keywords[:3])
            legend_items_html += (
                f'<div class="flex items-center gap-2">'
                f'<span class="w-2.5 h-2.5 rounded-full inline-block" style="background:{c.color};"></span>'
                f'<span class="text-xs text-gray-300"><b>{c.label}</b> ({c.size}) — {kw}</span>'
                f'</div>'
            )

        header_html = f"""
        <div class="bg-dark-800 p-4 m-3 rounded-xl border border-dark-600 font-mono text-gray-200">
          <div class="flex items-center gap-3 mb-3">
            <svg class="w-6 h-6 text-accent-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"/>
            </svg>
            <h2 class="text-lg font-bold text-accent-cyan">Research Map: {topic}</h2>
          </div>
          <div class="flex gap-4 text-xs text-gray-400 mb-3">
            <span><b class="text-gray-200">Papers:</b> {result.total_papers}</span>
            <span><b class="text-gray-200">Nodes:</b> {stats.node_count}</span>
            <span><b class="text-gray-200">Edges:</b> {result.edges}</span>
            <span><b class="text-gray-200">Clusters:</b> {len(clusters)}</span>
          </div>
          <hr class="border-dark-600 mb-3">
          <div class="space-y-1">{legend_items_html}</div>
          <hr class="border-dark-600 mt-3 mb-2">
          <p class="text-[10px] text-gray-500">Click any node to see full details &middot; Scroll to zoom &middot; ESC to close panel</p>
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
        net.heading = ""

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

        # Header + legend (Tailwind)
        legend_items_html = ""
        for c in result.clusters:
            kw = ", ".join(c.keywords)
            legend_items_html += (
                f'<div class="flex items-center gap-2">'
                f'<span class="w-2.5 h-2.5 rounded-full inline-block" style="background:{c.color};"></span>'
                f'<span class="text-xs text-gray-300"><b>{c.label}</b> ({c.size} cats) — {kw}</span>'
                f'</div>'
            )
        header_html = f"""
        <div class="bg-dark-800 p-4 m-3 rounded-xl border border-dark-600 font-mono text-gray-200">
          <div class="flex items-center gap-3 mb-3">
            <svg class="w-6 h-6 text-accent-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            <h2 class="text-lg font-bold text-accent-cyan">Research Landscape: {topic}</h2>
          </div>
          <div class="flex gap-4 text-xs text-gray-400 mb-3">
            <span><b class="text-gray-200">Papers:</b> {result.total_papers}</span>
            <span><b class="text-gray-200">Categories:</b> {len(all_cats)}</span>
            <span><b class="text-gray-200">Edges:</b> {result.edges}</span>
            <span><b class="text-gray-200">Clusters:</b> {len(result.clusters)}</span>
          </div>
          <hr class="border-dark-600 mb-3">
          <div class="space-y-1">{legend_items_html}</div>
          <hr class="border-dark-600 mt-3 mb-2">
          <div class="flex items-center gap-3 text-[10px] text-gray-500">
            <span>&#9679; Category (large)</span>
            <span>&#9670; Paper (small)</span>
            <span>Click for details</span>
            <span>ESC to close</span>
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
