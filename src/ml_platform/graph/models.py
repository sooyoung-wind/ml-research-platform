"""ML Research Platform — Knowledge Graph data models.

Defines node types, edge types, and graph statistics used
throughout the knowledge graph subsystem.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of entities that can be nodes in the knowledge graph.

    Attributes:
        PAPER: A research paper.
        AUTHOR: A researcher / author.
        METHOD: A methodology, algorithm, or technique.
        DATASET: A benchmark or dataset used for evaluation.
        CONCEPT: A domain concept or research area.
        INSTITUTION: A university or research lab.
        VENUE: A publication venue (conference, journal).
    """

    PAPER = "Paper"
    AUTHOR = "Author"
    METHOD = "Method"
    DATASET = "Dataset"
    CONCEPT = "Concept"
    INSTITUTION = "Institution"
    VENUE = "Venue"


class EdgeType(str, Enum):
    """Types of relationships between graph nodes.

    Attributes:
        CITES: Paper A cites Paper B.
        USES: Paper/Author uses a Method or Dataset.
        PROPOSES: Paper proposes a Method.
        EVALUATES_ON: Paper evaluates on a Dataset.
        AFFILIATED_WITH: Author affiliated with Institution.
        CONTRIBUTES_TO: Author contributes to Paper.
        BELONGS_TO: Paper belongs to Venue.
        RELATED_TO: Generic semantic relationship.
    """

    CITES = "CITES"
    USES = "USES"
    PROPOSES = "PROPOSES"
    EVALUATES_ON = "EVALUATES_ON"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    BELONGS_TO = "BELONGS_TO"
    RELATED_TO = "RELATED_TO"


class GraphNode(BaseModel):
    """A node in the knowledge graph.

    Attributes:
        node_id: Unique identifier (e.g. ``paper_2405_14831``).
        node_type: Type of entity this node represents.
        label: Human-readable label.
        properties: Additional key-value properties.
    """

    node_id: str
    node_type: NodeType
    label: str
    properties: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """A directed edge in the knowledge graph.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        edge_type: Type of relationship.
        properties: Additional key-value properties (weight, year, etc.).
    """

    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: dict = Field(default_factory=dict)


class GraphStats(BaseModel):
    """Statistics about a topic knowledge graph.

    Attributes:
        topic: The topic name.
        db_path: Path to the graph database file.
        node_count: Total number of nodes.
        edge_count: Total number of edges.
        node_types: Breakdown of node counts by type.
        edge_types: Breakdown of edge counts by type.
        papers_indexed: Number of papers in the graph.
        created_at: When the graph was created.
        updated_at: When the graph was last updated.
    """

    topic: str
    db_path: str
    node_count: int = 0
    edge_count: int = 0
    node_types: dict[str, int] = Field(default_factory=dict)
    edge_types: dict[str, int] = Field(default_factory=dict)
    papers_indexed: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
