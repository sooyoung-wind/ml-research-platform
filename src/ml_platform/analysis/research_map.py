"""ML Research Platform — Research Map module.

Generates interactive visual maps of research landscapes from
knowledge graphs, showing topic clusters, paper relationships,
and research frontiers.
"""

from __future__ import annotations

import json
import math
import os
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

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
    "#FF6B6B",  # red
    "#4ECDC4",  # teal
    "#45B7D1",  # sky blue
    "#96CEB4",  # sage
    "#FFEAA7",  # yellow
    "#DDA0DD",  # plum
    "#98D8C8",  # mint
    "#F7DC6F",  # gold
    "#BB8FCE",  # purple
    "#85C1E9",  # light blue
]

NODE_COLORS = {
    "paper": "#E74C3C",
    "author": "#3498DB",
    "method": "#2ECC71",
    "dataset": "#F39C12",
    "concept": "#9B59B6",
    "institution": "#1ABC9C",
    "venue": "#E67E22",
}


# ── Community detection (simple label propagation) ──────────────────


def _detect_communities(
    adjacency: dict[str, set[str]],
    node_labels: dict[str, str],
    max_iterations: int = 20,
) -> dict[str, int]:
    """Simple label-propagation community detection.

    Args:
        adjacency: node_id -> set of neighbor node_ids
        node_labels: node_id -> display label
        max_iterations: max propagation rounds

    Returns:
        node_id -> community_id mapping
    """
    nodes = list(adjacency.keys())
    if not nodes:
        return {}

    # Initialize: each node is its own community
    community = {n: i for i, n in enumerate(nodes)}

    for _ in range(max_iterations):
        changed = False
        for node in nodes:
            neighbors = adjacency.get(node, set())
            if not neighbors:
                continue

            # Count neighbor communities
            neighbor_comms = Counter(community.get(nb, 0) for nb in neighbors)
            best_comm = neighbor_comms.most_common(1)[0][0]

            if community[node] != best_comm:
                community[node] = best_comm
                changed = True

        if not changed:
            break

    # Renumber communities to 0..N
    unique_comms = sorted(set(community.values()))
    comm_map = {c: i for i, c in enumerate(unique_comms)}
    return {n: comm_map[community[n]] for n in nodes}


# ── Research Map Builder ────────────────────────────────────────────


class ResearchMapBuilder:
    """Builds interactive research landscape maps from knowledge graphs.

    Uses knowledge graph data to create pyvis network visualizations
    showing paper clusters, citation patterns, and research frontiers.
    """

    def __init__(self, data_dir: Path | str | None = None):
        if data_dir is None:
            data_dir = AppConfig.DATA_DIR / "maps"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def build_from_graph(
        self,
        topic: str,
        open_browser: bool = True,
        height: str = "900px",
        min_cluster_size: int = 1,
    ) -> ResearchMapResult:
        """Build research map from a knowledge graph.

        Args:
            topic: Knowledge graph topic name.
            open_browser: Whether to open the result in a browser.
            height: Visualization height.
            min_cluster_size: Minimum papers for a cluster to display.

        Returns:
            ResearchMapResult with cluster info and HTML path.
        """
        from ml_platform.graph.knowledge_graph import KnowledgeGraph

        result = ResearchMapResult(topic)

        # Load graph
        kg = KnowledgeGraph.open(topic)
        stats = kg.get_stats()

        if stats.node_count == 0:
            print(f"  [WARN] Knowledge graph '{topic}' is empty")
            return result

        # Fetch all nodes and edges via Cypher
        nodes_data = kg.query("MATCH (n) RETURN n.node_id, n.label, labels(n) as types")
        edges_data = kg.query(
            "MATCH (a)-[r]->(b) RETURN a.node_id, b.node_id, type(r) as rel_type"
        )

        kg.close()

        # Build node lookup
        node_map: dict[str, dict] = {}
        for row in nodes_data:
            nid = row.get("n.node_id", "")
            label = row.get("n.label", nid)
            types = row.get("types", [])
            node_map[nid] = {
                "id": nid,
                "label": label,
                "types": types,
            }

        # Build adjacency for community detection
        adjacency: dict[str, set[str]] = defaultdict(set)
        edge_list = []
        for row in edges_data:
            src = row.get("a.node_id", "")
            tgt = row.get("b.node_id", "")
            rel_type = row.get("rel_type", "RELATED")
            if src and tgt and src in node_map and tgt in node_map:
                adjacency[src].add(tgt)
                adjacency[tgt].add(src)
                edge_list.append((src, tgt, rel_type))

        # Detect communities
        communities = _detect_communities(adjacency, {n: d["label"] for n, d in node_map.items()})

        # Group nodes by community
        comm_nodes: dict[int, list[str]] = defaultdict(list)
        for nid, cid in communities.items():
            comm_nodes[cid].append(nid)

        # Build cluster info
        clusters: list[ClusterInfo] = []
        for cid in sorted(comm_nodes.keys()):
            nids = comm_nodes[cid]
            if len(nids) < min_cluster_size:
                continue

            # Determine cluster label from most common type + keywords
            type_counts: Counter = Counter()
            keywords: list[str] = []
            paper_ids: list[str] = []

            for nid in nids:
                node = node_map[nid]
                for t in node["types"]:
                    type_counts[t] += 1
                keywords.append(node["label"])
                if "Paper" in node["types"]:
                    paper_ids.append(nid)

            dominant_type = type_counts.most_common(1)[0][0] if type_counts else "Unknown"
            top_keywords = [k for k, _ in Counter(keywords).most_common(5)]

            cluster = ClusterInfo(cid, f"{dominant_type} Cluster {cid}")
            cluster.size = len(nids)
            cluster.color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            cluster.papers = paper_ids
            cluster.keywords = top_keywords[:5]
            clusters.append(cluster)

        result.clusters = clusters
        result.total_papers = stats.papers_indexed
        result.edges = len(edge_list)

        # ── Build pyvis network ──────────────────────────────────
        net = Network(
            height=height,
            width="100%",
            directed=False,
            notebook=False,
            bgcolor="#1a1a2e",
            font_color="#e0e0e0",
        )

        net.heading = f"Research Map: {topic}"

        # Add nodes
        for nid, node in node_map.items():
            comm_id = communities.get(nid, 0)
            cluster_color = CLUSTER_COLORS[comm_id % len(CLUSTER_COLORS)]

            # Node type determines shape
            types = node["types"]
            if "Paper" in types:
                shape = "dot"
                size = 20
                color = cluster_color
            elif "Author" in types:
                shape = "diamond"
                size = 12
                color = NODE_COLORS["author"]
            elif "Concept" in types or "Method" in types:
                shape = "triangle"
                size = 15
                color = cluster_color
            else:
                shape = "dot"
                size = 10
                color = cluster_color

            # Tooltip
            tooltip_parts = [f"<b>{node['label']}</b>"]
            tooltip_parts.append(f"Type: {', '.join(types)}")
            tooltip = "<br>".join(tooltip_parts)

            net.add_node(
                nid,
                label=node["label"][:30],
                title=tooltip,
                shape=shape,
                size=size,
                color=color,
                group=comm_id,
            )

        # Add edges
        for src, tgt, rel_type in edge_list:
            edge_color = "#333355"
            edge_width = 1
            if "CITES" in rel_type or "REFERENCES" in rel_type:
                edge_color = "#556677"
                edge_width = 2
            elif "CONTRIBUTES_TO" in rel_type:
                edge_color = "#445566"
                edge_width = 1

            net.add_edge(src, tgt, title=rel_type, color=edge_color, width=edge_width)

        # Physics layout
        net.force_atlas_2based(
            gravity=-80,
            central_gravity=0.01,
            spring_length=150,
            spring_strength=0.05,
            damping=0.4,
        )

        # Build cluster legend HTML
        legend_items = []
        for c in clusters:
            kw_str = ", ".join(c.keywords[:3])
            legend_items.append(
                f'<span style="color:{c.color};">&#9679;</span> '
                f"<b>{c.label}</b> ({c.size} nodes) — {kw_str}"
            )
        legend_html = "<br>".join(legend_items)

        # Stats footer
        stats_html = (
            f"<b>Papers:</b> {result.total_papers} | "
            f"<b>Nodes:</b> {stats.node_count} | "
            f"<b>Edges:</b> {result.edges} | "
            f"<b>Clusters:</b> {len(clusters)}"
        )

        # Inject custom HTML header
        custom_html = f"""
        <div style="background:#16213e;padding:15px;margin:10px;border-radius:8px;
                    color:#e0e0e0;font-family:monospace;">
            <h2 style="margin:0 0 10px 0;color:#4ECDC4;">
                Research Map: {topic}
            </h2>
            <div style="font-size:12px;">
                {stats_html}
            </div>
            <hr style="border-color:#333;">
            <div style="font-size:11px;line-height:1.6;">
                {legend_html}
            </div>
            <hr style="border-color:#333;">
            <div style="font-size:10px;color:#888;">
                Node shapes: ● Paper ◆ Author ▲ Concept/Method | Scroll to zoom, drag to move
            </div>
        </div>
        """

        # Save
        output_path = self.data_dir / f"map_{topic}.html"
        net.save_graph(str(output_path))

        # Inject custom header into saved HTML
        html_content = output_path.read_text()
        html_content = html_content.replace(
            "<body>",
            f"<body>\n{custom_html}",
        )
        output_path.write_text(html_content)

        result.html_path = str(output_path)

        if open_browser:
            webbrowser.open(f"file://{output_path.resolve()}")

        return result

    def build_from_papers(
        self,
        topic: str,
        papers: list | None = None,
        open_browser: bool = True,
    ) -> ResearchMapResult:
        """Build research map directly from papers in DB.

        Creates a co-category / co-keyword network when no knowledge
        graph exists yet.
        """
        from ml_platform.db import PapersDB

        result = ResearchMapResult(topic)

        db = PapersDB()
        all_papers = papers or db.get_papers(limit=1000)
        result.total_papers = len(all_papers)

        if not all_papers:
            return result

        # Build co-category network
        cat_papers: dict[str, list[str]] = defaultdict(list)
        paper_cats: dict[str, list[str]] = {}

        for paper in all_papers:
            pid = paper.paper_id
            cats = paper.categories if paper.categories else ["uncategorized"]
            paper_cats[pid] = cats
            for cat in cats:
                cat_papers[cat].append(pid)

        # Build adjacency between categories (co-occurrence)
        cat_adjacency: dict[str, set[str]] = defaultdict(set)
        edge_set: set[tuple[str, str]] = set()

        for pid, cats in paper_cats.items():
            for i, c1 in enumerate(cats):
                for c2 in cats[i + 1:]:
                    pair = tuple(sorted([c1, c2]))
                    if pair not in edge_set:
                        edge_set.add(pair)
                        cat_adjacency[c1].add(c2)
                        cat_adjacency[c2].add(c1)

        # Detect communities among categories
        all_cats = list(cat_papers.keys())
        cat_communities = _detect_communities(
            cat_adjacency,
            {c: c for c in all_cats},
        )

        # ── Build pyvis network ──
        net = Network(
            height="900px",
            width="100%",
            directed=False,
            notebook=False,
            bgcolor="#1a1a2e",
            font_color="#e0e0e0",
        )

        net.heading = f"Research Landscape: {topic}"

        # Add category nodes
        for cat in all_cats:
            comm_id = cat_communities.get(cat, 0)
            color = CLUSTER_COLORS[comm_id % len(CLUSTER_COLORS)]
            paper_count = len(cat_papers[cat])

            # Year distribution
            years: list[int] = []
            for p in all_papers:
                if cat in paper_cats.get(p.paper_id, []):
                    if p.year:
                        years.append(p.year)

            year_str = f"{min(years)}-{max(years)}" if years else "N/A"
            tooltip = (
                f"<b>{cat}</b><br>"
                f"Papers: {paper_count}<br>"
                f"Period: {year_str}"
            )

            net.add_node(
                cat,
                label=cat,
                title=tooltip,
                shape="dot",
                size=min(15 + paper_count * 3, 50),
                color=color,
                group=comm_id,
            )

        # Add co-occurrence edges
        for c1, c2 in edge_set:
            # Weight = number of shared papers
            shared = len(set(cat_papers[c1]) & set(cat_papers[c2]))
            net.add_edge(c1, c2, title=f"shared: {shared}", width=min(shared, 5), color="#334455")

        result.edges = len(edge_set)

        # Build clusters
        comm_cats: dict[int, list[str]] = defaultdict(list)
        for cat, cid in cat_communities.items():
            comm_cats[cid].append(cat)

        for cid in sorted(comm_cats.keys()):
            cats_in_comm = comm_cats[cid]
            total_papers_in_comm = sum(len(cat_papers[c]) for c in cats_in_comm)
            cluster = ClusterInfo(cid, f"Cluster {cid}: {cats_in_comm[0]}")
            cluster.size = len(cats_in_comm)
            cluster.color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            cluster.papers = [str(p) for c in cats_in_comm for p in cat_papers[c][:3]]
            cluster.keywords = cats_in_comm[:5]
            result.clusters.append(cluster)

        # Legend
        legend_items = []
        for c in result.clusters:
            kw_str = ", ".join(c.keywords)
            legend_items.append(
                f'<span style="color:{c.color};">&#9679;</span> '
                f"<b>{c.label}</b> ({c.size} categories) — {kw_str}"
            )

        custom_html = f"""
        <div style="background:#16213e;padding:15px;margin:10px;border-radius:8px;
                    color:#e0e0e0;font-family:monospace;">
            <h2 style="margin:0 0 10px 0;color:#4ECDC4;">
                Research Landscape: {topic}
            </h2>
            <div style="font-size:12px;">
                <b>Papers:</b> {result.total_papers} |
                <b>Categories:</b> {len(all_cats)} |
                <b>Co-occurrences:</b> {result.edges} |
                <b>Clusters:</b> {len(result.clusters)}
            </div>
            <hr style="border-color:#333;">
            <div style="font-size:11px;line-height:1.6;">
                {"<br>".join(legend_items)}
            </div>
            <hr style="border-color:#333;">
            <div style="font-size:10px;color:#888;">
                Node size = paper count | Edge width = shared papers | Scroll to zoom
            </div>
        </div>
        """

        output_path = self.data_dir / f"map_{topic}.html"
        net.save_graph(str(output_path))

        html_content = output_path.read_text()
        html_content = html_content.replace("<body>", f"<body>\n{custom_html}")
        output_path.write_text(html_content)

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
