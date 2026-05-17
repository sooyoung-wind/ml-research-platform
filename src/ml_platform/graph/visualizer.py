"""ML Research Platform — Graph visualizer.

Generates interactive HTML visualizations of knowledge graphs
using pyvis (NetworkX + vis.js).
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

from pyvis.network import Network

from ml_platform.graph.knowledge_graph import KnowledgeGraph
from ml_platform.graph.models import NodeType


# Color scheme per node type
NODE_COLORS: dict[str, str] = {
    NodeType.PAPER.value: "#FF6B6B",       # coral red
    NodeType.AUTHOR.value: "#4ECDC4",      # teal
    NodeType.METHOD.value: "#45B7D1",      # sky blue
    NodeType.DATASET.value: "#96CEB4",     # sage green
    NodeType.CONCEPT.value: "#FFEAA7",     # pale yellow
    NodeType.INSTITUTION.value: "#DDA0DD", # plum
    NodeType.VENUE.value: "#98D8C8",       # mint
}

NODE_SIZES: dict[str, int] = {
    NodeType.PAPER.value: 30,
    NodeType.AUTHOR.value: 20,
    NodeType.METHOD.value: 25,
    NodeType.DATASET.value: 20,
    NodeType.CONCEPT.value: 22,
    NodeType.INSTITUTION.value: 18,
    NodeType.VENUE.value: 18,
}

EDGE_COLORS: dict[str, str] = {
    "CITES": "#FF6B6B",
    "USES": "#45B7D1",
    "PROPOSES": "#96CEB4",
    "EVALUATES_ON": "#FFEAA7",
    "AFFILIATED_WITH": "#DDA0DD",
    "CONTRIBUTES_TO": "#4ECDC4",
    "BELONGS_TO": "#98D8C8",
    "RELATED_TO": "#CCCCCC",
}


def visualize_graph(
    topic: str,
    output_path: str | Path | None = None,
    base_dir: str | Path | None = None,
    open_browser: bool = True,
    height: str = "800px",
    width: str = "100%",
    physics: bool = True,
) -> Path:
    """Generate an interactive HTML visualization of a knowledge graph.

    Args:
        topic: Topic graph name.
        output_path: Output HTML file path. Defaults to ``data/graphs/viz_{topic}.html``.
        base_dir: Graph database directory.
        open_browser: Whether to open the HTML in a browser.
        height: Visualization height.
        width: Visualization width.
        physics: Whether to enable physics simulation.

    Returns:
        Path to the generated HTML file.
    """
    kg = KnowledgeGraph.open(topic, base_dir=base_dir)
    try:
        stats = kg.get_stats()
        nodes = kg.get_all_nodes()
        edges = kg.get_all_edges()

        if not nodes:
            raise ValueError(f"Graph '{topic}' is empty — no nodes found.")

        # Create pyvis network
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=False,
            bgcolor="#1a1a2e",
            font_color="white",
        )

        # Physics settings
        if physics:
            net.barnes_hut(
                gravity=-5000,
                central_gravity=0.3,
                spring_length=150,
                spring_strength=0.001,
                damping=0.09,
            )

        # Add nodes
        for node in nodes:
            node_id = node.get("n.node_id", str(node))
            label = node.get("n.label", node_id)

            # Determine type
            types = node.get("types", [])
            if isinstance(types, list) and types:
                ntype = types[0]
            else:
                ntype = str(types)

            # Truncate label for display
            display_label = label[:50] + "..." if len(label) > 50 else label

            color = NODE_COLORS.get(ntype, "#CCCCCC")
            size = NODE_SIZES.get(ntype, 15)

            # Build tooltip
            tooltip = f"[{ntype}] {label}"
            if node_id:
                tooltip += f"\nID: {node_id}"

            net.add_node(
                node_id,
                label=display_label,
                title=tooltip,
                color=color,
                size=size,
                shape="dot" if ntype != NodeType.PAPER.value else "diamond",
                borderWidth=2,
                borderWidthSelected=4,
            )

        # Add edges
        for edge in edges:
            src = edge.get("a.node_id", "")
            tgt = edge.get("b.node_id", "")
            rel = edge.get("rel", "RELATED_TO")

            color = EDGE_COLORS.get(rel, "#888888")

            net.add_edge(
                src, tgt,
                label=rel,
                color=color,
                arrows="to",
                title=rel,
                width=1.5,
                smooth={"type": "curvedCW", "roundness": 0.2},
            )

        # Legend as HTML
        legend_items = []
        for ntype, color in NODE_COLORS.items():
            count = stats.node_types.get(ntype, 0)
            if count > 0:
                legend_items.append(
                    f'<span style="color:{color}; font-weight:bold;">●</span> '
                    f"{ntype} ({count})"
                )
        legend_html = "<br>".join(legend_items)

        # Set options
        net.set_options(f"""
        {{
            "nodes": {{
                "font": {{"size": 12, "color": "white"}},
                "scaling": {{}}
            }},
            "edges": {{
                "font": {{"size": 9, "color": "#aaaaaa", "strokeWidth": 0}},
                "smooth": {{"type": "curvedCW", "roundness": 0.2}}
            }},
            "interaction": {{
                "hover": true,
                "tooltipDelay": 200,
                "navigationButtons": true,
                "keyboard": true
            }},
            "legend": {{
                "useLogo": false
            }}
        }}
        """)

        # Output path
        if output_path is None:
            out = Path("data/graphs") / f"viz_{topic.replace(' ', '_')}.html"
        else:
            out = Path(output_path)

        out.parent.mkdir(parents=True, exist_ok=True)

        # Save and inject legend + title
        net.save_graph(str(out))
        _inject_header(out, topic, stats, legend_html)

        print(f"[INFO] Graph visualization saved to {out}")
        print(f"  Nodes: {stats.node_count} | Edges: {stats.edge_count}")

        if open_browser:
            webbrowser.open(f"file://{out.resolve()}")
            print(f"[INFO] Opened in browser.")

        return out

    finally:
        kg.close()


def _inject_header(html_path: Path, topic: str, stats, legend: str) -> None:
    """Inject a custom header with legend into the HTML file.

    Args:
        html_path: Path to the generated HTML.
        topic: Topic name.
        stats: GraphStats object.
        legend: HTML string for the legend.
    """
    content = html_path.read_text()

    header = f"""
    <div id="graph-header" style="
        position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
        background: rgba(26,26,46,0.95); color: white;
        padding: 12px 20px; font-family: -apple-system, sans-serif;
        border-bottom: 2px solid #4ECDC4;
        display: flex; justify-content: space-between; align-items: center;
    ">
        <div>
            <h2 style="margin:0; color:#4ECDC4;">Knowledge Graph: {topic}</h2>
            <p style="margin:2px 0 0 0; color:#aaa; font-size:13px;">
                {stats.node_count} nodes | {stats.edge_count} edges | {stats.papers_indexed} papers
            </p>
        </div>
        <div style="font-size:13px; line-height:1.6; text-align:right;">
            {legend}
        </div>
    </div>
    """

    # Inject before the vis network div
    content = content.replace("<body>", f"<body>\n{header}", 1)

    # Push vis div down to account for header
    content = content.replace(
        'id="mynetwork"',
        'id="mynetwork" style="margin-top: 90px;"',
    )

    html_path.write_text(content)
