"""ML Research Platform — Graph Builder orchestrator.

Builds per-topic knowledge graphs from paper analyses by extracting
entities and relationships, then persisting them via KnowledgeGraph.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ml_platform.analysis.models import PaperAnalysis
from ml_platform.db import PapersDB
from ml_platform.graph.entity_resolver import extract_edges, extract_entities
from ml_platform.graph.knowledge_graph import KnowledgeGraph
from ml_platform.graph.models import EdgeType, GraphStats, NodeType


class GraphBuilder:
    """Orchestrates building knowledge graphs from paper analyses.

    Fetches analyses from the database, extracts entities/edges,
    resolves duplicates, and populates a per-topic graph.

    Usage::

        builder = GraphBuilder()
        stats = builder.build_topic("retrieval_augmented_generation")
    """

    def __init__(self, db: PapersDB | None = None) -> None:
        """Initialize the builder.

        Args:
            db: PapersDB instance. Creates a default one if not provided.
        """
        self.db = db or PapersDB()

    def build_topic(
        self,
        topic: str,
        paper_ids: list[str] | None = None,
        base_dir: str | Path | None = None,
        force: bool = False,
    ) -> GraphStats:
        """Build a knowledge graph for a topic.

        Args:
            topic: Topic name (e.g. ``"diffusion_models"``).
            paper_ids: Optional list of paper IDs to include.
                If None, uses all analyzed papers in the DB.
            base_dir: Directory for graph databases.
            force: If True, rebuild from scratch even if graph exists.

        Returns:
            GraphStats for the built graph.
        """
        start = time.time()
        kg = KnowledgeGraph.open(topic, base_dir=base_dir)

        try:
            # Get analyses
            analyses = self._fetch_analyses(paper_ids)
            if not analyses:
                print(f"[WARN] No analyses found for topic '{topic}'")
                return kg.get_stats()

            print(f"[INFO] Building graph '{topic}' from {len(analyses)} analyses...")

            total_nodes = 0
            total_edges = 0

            for i, analysis in enumerate(analyses, 1):
                pid = analysis.paper_id
                print(f"  [{i}/{len(analyses)}] Processing {pid}...")

                # Extract entities
                nodes = extract_entities(analysis, paper_id=pid)
                if not nodes:
                    continue

                # Get references for citation edges
                references = self._get_references(analysis)

                # Extract edges
                edges = extract_edges(analysis, nodes, references=references)

                # Add nodes to graph
                for node in nodes:
                    kg.add_node(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        label=node.label,
                        **node.properties,
                    )

                # Add edges to graph
                for edge in edges:
                    kg.add_edge(
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        edge_type=edge.edge_type,
                        **edge.properties,
                    )

                total_nodes += len(nodes)
                total_edges += len(edges)

            stats = kg.get_stats()
            elapsed = time.time() - start
            print(
                f"[INFO] Graph '{topic}' built: {stats.node_count} nodes, "
                f"{stats.edge_count} edges ({elapsed:.1f}s)"
            )
            return stats

        finally:
            kg.close()

    def add_paper(
        self,
        paper_id: str,
        topic: str,
        source: str = "arxiv",
        base_dir: str | Path | None = None,
    ) -> GraphStats:
        """Add a single paper to an existing topic graph.

        Args:
            paper_id: Paper identifier.
            topic: Topic graph name.
            source: Paper source.
            base_dir: Graph database directory.

        Returns:
            Updated GraphStats.

        Raises:
            ValueError: If no analysis exists for the paper.
        """
        analysis = self.db.get_analysis_object(paper_id, source)
        if analysis is None:
            raise ValueError(f"No analysis found for {paper_id} (source={source})")

        analysis.paper_id = paper_id

        kg = KnowledgeGraph.open(topic, base_dir=base_dir)
        try:
            nodes = extract_entities(analysis, paper_id=paper_id)
            references = self._get_references(analysis)
            edges = extract_edges(analysis, nodes, references=references)

            for node in nodes:
                kg.add_node(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    label=node.label,
                    **node.properties,
                )

            for edge in edges:
                kg.add_edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_type=edge.edge_type,
                    **edge.properties,
                )

            return kg.get_stats()
        finally:
            kg.close()

    def remove_paper(
        self,
        paper_id: str,
        topic: str,
        base_dir: str | Path | None = None,
    ) -> GraphStats:
        """Remove a paper and its author/method edges from a topic graph.

        Note: Does not remove shared entities (methods, datasets) that
        may be referenced by other papers.

        Args:
            paper_id: Paper identifier.
            topic: Topic graph name.
            base_dir: Graph database directory.

        Returns:
            Updated GraphStats.
        """
        pid_node = f"paper_{paper_id.replace('.', '_').replace('/', '_')}"
        kg = KnowledgeGraph.open(topic, base_dir=base_dir)
        try:
            # Remove all edges involving this paper
            kg.query(f"MATCH (p {{node_id: '{pid_node}'}})-[r]-() DELETE r")
            kg.query(f"MATCH ()-[r]-(p {{node_id: '{pid_node}'}}) DELETE r")
            # Remove the paper node
            kg.query(f"MATCH (p {{node_id: '{pid_node}'}}) DELETE p")
            return kg.get_stats()
        finally:
            kg.close()

    @staticmethod
    def list_graphs(base_dir: str | Path | None = None) -> list[dict]:
        """List all topic graphs with their stats.

        Args:
            base_dir: Graph database directory.

        Returns:
            List of dicts with topic, path, and stats.
        """
        base = Path(base_dir) if base_dir else Path("data/graphs")
        if not base.exists():
            return []

        results = []
        for db_file in sorted(base.glob("graph_*.db")):
            topic = db_file.stem.replace("graph_", "").replace("_", " ").title()
            try:
                kg = KnowledgeGraph.open(topic, base_dir=base_dir)
                stats = kg.get_stats()
                kg.close()
                results.append({
                    "topic": topic,
                    "db_path": str(db_file),
                    "stats": stats.model_dump(),
                })
            except Exception as e:
                results.append({
                    "topic": topic,
                    "db_path": str(db_file),
                    "error": str(e),
                })
        return results

    # ── Internal helpers ─────────────────────────────────────────────

    def _fetch_analyses(
        self, paper_ids: list[str] | None = None
    ) -> list[PaperAnalysis]:
        """Fetch paper analyses from the database.

        Args:
            paper_ids: Optional filter by paper IDs.

        Returns:
            List of PaperAnalysis objects.
        """
        analyses: list[PaperAnalysis] = []

        if paper_ids:
            for pid in paper_ids:
                for source in ["arxiv", "semantic_scholar", "huggingface"]:
                    analysis = self.db.get_analysis_object(pid, source)
                    if analysis is not None:
                        analysis.paper_id = pid
                        analyses.append(analysis)
                        break
        else:
            # Get all analyzed papers
            with self.db._conn() as conn:
                rows = conn.execute(
                    "SELECT paper_id, source, analysis_json "
                    "FROM analysis_results WHERE status = 'completed'"
                ).fetchall()
                for row in rows:
                    try:
                        from ml_platform.analysis.models import PaperAnalysis
                        analysis = PaperAnalysis.model_validate_json(
                            row["analysis_json"]
                        )
                        analysis.paper_id = row["paper_id"]
                        analyses.append(analysis)
                    except Exception:
                        continue

        return analyses

    def _get_references(self, analysis: PaperAnalysis) -> list[dict]:
        """Extract reference info from analysis for citation edges.

        Args:
            analysis: PaperAnalysis with references.

        Returns:
            List of reference dicts with paper_id.
        """
        refs = []
        if not analysis.references:
            return refs

        for ref in analysis.references:
            ref_dict: dict = {}
            if isinstance(ref, dict):
                ref_dict = ref
            elif hasattr(ref, "model_dump"):
                ref_dict = ref.model_dump()
            else:
                continue

            # Try to extract a paper_id
            pid = (
                ref_dict.get("arxiv_id")
                or ref_dict.get("paper_id")
                or ref_dict.get("doi")
            )
            if pid:
                refs.append({
                    "paper_id": pid,
                    "year": ref_dict.get("year", 0),
                })

        return refs
