"""ML Research Platform — Knowledge Graph core.

Wraps graphqlite to provide per-topic knowledge graphs stored as
SQLite databases with Cypher query support.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import graphqlite

from ml_platform.graph.models import EdgeType, GraphStats, NodeType


class KnowledgeGraph:
    """Per-topic knowledge graph backed by graphqlite (SQLite + Cypher).

    Each topic gets its own database file under ``data/graphs/``.
    Nodes represent entities (Paper, Author, Method, etc.) and edges
    represent relationships (CITES, USES, PROPOSES, etc.).

    Usage::

        kg = KnowledgeGraph.open("diffusion_models")
        result = kg.query("MATCH (p:Paper) RETURN p.title")
        kg.close()
    """

    def __init__(self, db_path: str | Path, topic: str) -> None:
        """Initialize graph from an existing database.

        Use :meth:`open` to create/open a graph by topic name.

        Args:
            db_path: Path to the graphqlite database file.
            topic: Topic name for this graph.
        """
        self.db_path = Path(db_path)
        self.topic = topic
        self._graph: graphqlite.Graph | None = None

    @classmethod
    def open(cls, topic: str, base_dir: str | Path | None = None) -> KnowledgeGraph:
        """Open or create a knowledge graph for a topic.

        Args:
            topic: Topic name (e.g. ``"diffusion_models"``).
            base_dir: Directory for graph databases. Defaults to ``data/graphs/``.

        Returns:
            An opened KnowledgeGraph instance.
        """
        base = Path(base_dir) if base_dir else Path("data/graphs")
        base.mkdir(parents=True, exist_ok=True)

        safe_name = topic.replace(" ", "_").replace("/", "_").lower()
        db_path = base / f"graph_{safe_name}.db"

        kg = cls(db_path, topic)
        kg._graph = graphqlite.Graph(str(db_path))
        return kg

    @property
    def graph(self) -> graphqlite.Graph:
        """The underlying graphqlite Graph instance.

        Returns:
            The graphqlite Graph.

        Raises:
            RuntimeError: If the graph has not been opened.
        """
        if self._graph is None:
            raise RuntimeError("Graph not opened. Use KnowledgeGraph.open().")
        return self._graph

    def close(self) -> None:
        """Close the graph database connection."""
        self._graph = None

    def __enter__(self) -> KnowledgeGraph:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Node operations ──────────────────────────────────────────────

    def add_node(
        self,
        node_id: str,
        node_type: NodeType | str,
        label: str,
        **properties: Any,
    ) -> None:
        """Add or update a node in the graph.

        Uses Cypher MERGE to avoid duplicates.

        Args:
            node_id: Unique node identifier.
            node_type: Node type label (e.g. ``NodeType.PAPER``).
            label: Human-readable label.
            **properties: Additional properties stored on the node.
        """
        ntype = node_type.value if isinstance(node_type, NodeType) else node_type
        props_str = self._format_props(
            node_id=node_id, label=label, **properties
        )
        cypher = f"MERGE (n:{ntype} {{{props_str}}})"
        self.graph.query(cypher)

    def add_paper_node(
        self,
        paper_id: str,
        title: str,
        year: int | None = None,
        **extra: Any,
    ) -> None:
        """Add a Paper node.

        Args:
            paper_id: Paper identifier (e.g. arXiv ID).
            title: Paper title.
            year: Publication year.
            **extra: Additional properties.
        """
        self.add_node(
            node_id=f"paper_{paper_id.replace('.', '_').replace('/', '_')}",
            node_type=NodeType.PAPER,
            label=title,
            paper_id=paper_id,
            year=year or 0,
            **extra,
        )

    def add_author_node(self, name: str, **extra: Any) -> None:
        """Add an Author node.

        Args:
            name: Author name (used as both ID and label).
            **extra: Additional properties.
        """
        node_id = f"author_{self._slugify(name)}"
        self.add_node(
            node_id=node_id,
            node_type=NodeType.AUTHOR,
            label=name,
            **extra,
        )

    def add_method_node(self, name: str, **extra: Any) -> None:
        """Add a Method node.

        Args:
            name: Method name (e.g. ``"PPR"``).
            **extra: Additional properties.
        """
        node_id = f"method_{self._slugify(name)}"
        self.add_node(
            node_id=node_id,
            node_type=NodeType.METHOD,
            label=name,
            **extra,
        )

    def add_dataset_node(self, name: str, **extra: Any) -> None:
        """Add a Dataset node.

        Args:
            name: Dataset name.
            **extra: Additional properties.
        """
        node_id = f"dataset_{self._slugify(name)}"
        self.add_node(
            node_id=node_id,
            node_type=NodeType.DATASET,
            label=name,
            **extra,
        )

    def add_concept_node(self, name: str, **extra: Any) -> None:
        """Add a Concept node.

        Args:
            name: Concept name.
            **extra: Additional properties.
        """
        node_id = f"concept_{self._slugify(name)}"
        self.add_node(
            node_id=node_id,
            node_type=NodeType.CONCEPT,
            label=name,
            **extra,
        )

    def add_institution_node(self, name: str, **extra: Any) -> None:
        """Add an Institution node.

        Args:
            name: Institution name.
            **extra: Additional properties.
        """
        node_id = f"inst_{self._slugify(name)}"
        self.add_node(
            node_id=node_id,
            node_type=NodeType.INSTITUTION,
            label=name,
            **extra,
        )

    def add_venue_node(self, name: str, **extra: Any) -> None:
        """Add a Venue node.

        Args:
            name: Venue name.
            **extra: Additional properties.
        """
        node_id = f"venue_{self._slugify(name)}"
        self.add_node(
            node_id=node_id,
            node_type=NodeType.VENUE,
            label=name,
            **extra,
        )

    # ── Edge operations ──────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType | str,
        **properties: Any,
    ) -> None:
        """Add a directed edge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Relationship type.
            **properties: Additional edge properties.
        """
        etype = edge_type.value if isinstance(edge_type, EdgeType) else edge_type
        props_str = ""
        if properties:
            props_str = " {" + self._dict_to_cypher(properties) + "}"

        cypher = (
            f"MATCH (a {{node_id: '{source_id}'}}), "
            f"(b {{node_id: '{target_id}'}}) "
            f"MERGE (a)-[:{etype}{props_str}]->(b)"
        )
        self.graph.query(cypher)

    # ── Query operations ─────────────────────────────────────────────

    def query(self, cypher: str) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results.

        Args:
            cypher: Cypher query string.

        Returns:
            List of result dictionaries.
        """
        raw = self.graph.query(cypher)
        if not raw:
            return []
        if isinstance(raw, list):
            return [self._parse_result(r) for r in raw]
        return [self._parse_result(raw)]

    def get_stats(self) -> GraphStats:
        """Compute statistics about the graph.

        Returns:
            GraphStats with node/edge counts and breakdowns.
        """
        node_types: dict[str, int] = {}
        edge_types: dict[str, int] = {}
        total_nodes = 0
        total_edges = 0

        try:
            for nt in NodeType:
                result = self.graph.query(f"MATCH (n:{nt.value}) RETURN count(n) as cnt")
                cnt = self._extract_count(result)
                if cnt > 0:
                    node_types[nt.value] = cnt
                    total_nodes += cnt

            for et in EdgeType:
                result = self.graph.query(
                    f"MATCH ()-[r:{et.value}]->() RETURN count(r) as cnt"
                )
                cnt = self._extract_count(result)
                if cnt > 0:
                    edge_types[et.value] = cnt
                    total_edges += cnt
        except Exception:
            pass

        papers = node_types.get("Paper", 0)

        return GraphStats(
            topic=self.topic,
            db_path=str(self.db_path),
            node_count=total_nodes,
            edge_count=total_edges,
            node_types=node_types,
            edge_types=edge_types,
            papers_indexed=papers,
        )

    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Get all nodes in the graph.

        Returns:
            List of node dictionaries.
        """
        return self.query("MATCH (n) RETURN n.node_id, n.label, labels(n) as types")

    def get_all_edges(self) -> list[dict[str, Any]]:
        """Get all edges in the graph.

        Returns:
            List of edge dictionaries.
        """
        return self.query(
            "MATCH (a)-[r]->(b) RETURN a.node_id, type(r) as rel, b.node_id"
        )

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a safe identifier slug.

        Args:
            text: Input text.

        Returns:
            Lowercase, underscore-separated slug.
        """
        slug = text.lower().strip()
        slug = "".join(c if c.isalnum() else "_" for c in slug)
        slug = "_".join(part for part in slug.split("_") if part)
        return slug[:80]

    @staticmethod
    def _format_props(**kwargs: Any) -> str:
        """Format keyword arguments as Cypher property string.

        Args:
            **kwargs: Properties to format.

        Returns:
            Cypher property string like ``key1: 'val1', key2: 42``.
        """
        parts = []
        for k, v in kwargs.items():
            if isinstance(v, str):
                escaped = v.replace("'", "\\'").replace('"', '\\"')
                parts.append(f"{k}: '{escaped}'")
            elif isinstance(v, bool):
                parts.append(f"{k}: {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                parts.append(f"{k}: {v}")
            elif v is None:
                continue
            else:
                escaped = json.dumps(v, ensure_ascii=False).replace("'", "\\'")
                parts.append(f"{k}: '{escaped}'")
        return ", ".join(parts)

    @staticmethod
    def _dict_to_cypher(d: dict[str, Any]) -> str:
        """Convert a dict to Cypher property string.

        Args:
            d: Properties dictionary.

        Returns:
            Cypher property string.
        """
        return KnowledgeGraph._format_props(**d)

    @staticmethod
    def _extract_count(result: Any) -> int:
        """Extract a count value from a Cypher query result.

        Args:
            result: Raw graphqlite query result.

        Returns:
            Integer count, or 0 if extraction fails.
        """
        try:
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict):
                    return int(first.get("cnt", 0))
                if isinstance(first, (int, float)):
                    return int(first)
            if isinstance(result, dict):
                return int(result.get("cnt", 0))
            if isinstance(result, (int, float)):
                return int(result)
        except (ValueError, TypeError, IndexError):
            pass
        return 0

    @staticmethod
    def _parse_result(r: Any) -> dict[str, Any]:
        """Parse a single graphqlite result into a dict.

        Args:
            r: Raw result from graphqlite.

        Returns:
            Dictionary representation.
        """
        if isinstance(r, dict):
            return r
        if hasattr(r, "__dict__"):
            return r.__dict__
        return {"value": r}
