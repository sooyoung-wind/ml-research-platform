# Phase 6 — Knowledge Graph Builder Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a graphqlite-based Knowledge Graph system that transforms paper analyses into queryable per-topic graphs with Cypher support.

**Architecture:** Each research topic gets its own SQLite+graphqlite database (`graph_{topic}.db`). The GraphBuilder extracts entities (Authors, Methods, Datasets, Concepts, Institutions) from PaperAnalysis results, resolves them across papers, and creates nodes/edges using Cypher queries. Users query via `ml-research graph` CLI commands.

**Tech Stack:** graphqlite 0.4.4 (SQLite + Cypher), Pydantic models, Typer CLI

**Existing assets:**
- `entity_registry` table in PapersDB (entity resolution foundation)
- `PaperAnalysis` model with 5W1H, references, evidence
- `reference_chain.py` with `canonical_paper_id()` for entity resolution

---

### Task 1: Create graph module structure + models

**Files:**
- Create: `src/ml_platform/graph/__init__.py`
- Create: `src/ml_platform/graph/models.py`

**Implementation:**
```python
# models.py — Graph node/edge type definitions
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class NodeType(str, Enum):
    PAPER = "Paper"
    AUTHOR = "Author"
    METHOD = "Method"
    DATASET = "Dataset"
    CONCEPT = "Concept"
    INSTITUTION = "Institution"
    VENUE = "Venue"

class EdgeType(str, Enum):
    CITES = "CITES"
    USES = "USES"
    PROPOSES = "PROPOSES"
    EVALUATES_ON = "EVALUATES_ON"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    BELONGS_TO = "BELONGS_TO"
    RELATED_TO = "RELATED_TO"

class GraphNode(BaseModel):
    node_id: str
    node_type: NodeType
    label: str
    properties: dict = Field(default_factory=dict)

class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: dict = Field(default_factory=dict)

class GraphStats(BaseModel):
    topic: str
    db_path: str
    node_count: int = 0
    edge_count: int = 0
    node_types: dict[str, int] = Field(default_factory=dict)
    edge_types: dict[str, int] = Field(default_factory=dict)
    papers_indexed: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
```

### Task 2: Create KnowledgeGraph core (open/close/query)

**Files:**
- Create: `src/ml_platform/graph/knowledge_graph.py`

**Implementation:** KnowledgeGraph class wrapping graphqlite
- `open(topic)` — opens/creates `data/graphs/graph_{topic}.db`
- `close()` — cleanup
- `execute_cypher(query, params)` — run arbitrary Cypher
- `get_stats()` — GraphStats
- Schema init: ensure indexes exist

### Task 3: Create EntityResolver

**Files:**
- Create: `src/ml_platform/graph/entity_resolver.py`

**Implementation:** Extract entities from PaperAnalysis
- `extract_entities(analysis, paper) -> list[GraphNode]`
  - Paper node (always)
  - Author nodes from analysis.five_w1h.who
  - Method nodes from analysis.five_w1h.how
  - Dataset nodes from analysis.five_w1h.how (dataset mentions)
  - Concept nodes from analysis.domain + analysis.key_contributions
  - Institution nodes from analysis.five_w1h.who
  - Venue node from analysis.five_w1h.where
- `extract_edges(analysis, entities) -> list[GraphEdge]`
  - CITES edges from analysis.references
  - USES edges (method → dataset)
  - PROPOSES edges (paper → method)
  - AFFILIATED_WITH (author → institution)
  - CONTRIBUTES_TO (author → paper)
- Resolve duplicates via entity_registry in PapersDB

### Task 4: Create GraphBuilder orchestrator

**Files:**
- Create: `src/ml_platform/graph/builder.py`

**Implementation:** Build graph from paper analyses
- `build_for_topic(topic, paper_ids=None)` — build graph for a topic
  - Fetch analyses from PapersDB
  - For each analysis: extract entities + edges → merge into graph
  - Entity resolution: deduplicate nodes across papers
- `add_paper(paper_id, topic)` — add single paper to existing graph
- `remove_paper(paper_id, topic)` — remove paper from graph
- `merge_topics(topic_a, topic_b, new_topic)` — merge two graphs

### Task 5: Create Cypher query presets

**Files:**
- Create: `src/ml_platform/graph/queries.py`

**Implementation:** Pre-built useful queries
- `papers_by_method(method_name)` — find papers using a method
- `citation_chain(paper_id, depth=3)` — BFS citation chain
- `author_collaboration(author_name)` — co-author network
- `method_evolution()` — timeline of methods used
- `research_gaps()` — find under-explored areas
- `trending_topics(top_n=10)` — most connected concepts

### Task 6: CLI integration

**Files:**
- Modify: `src/ml_platform/cli.py`

**Implementation:** Add `graph` command group
- `ml-research graph build <topic> [--papers <ids>] [--force]`
- `ml-research graph query <topic> <cypher>`
- `ml-research graph stats [<topic>]`
- `ml-research graph merge <topic_a> <topic_b> [--name <new>]`
- `ml-research graph list` — list all topic graphs
- `ml-research graph export <topic> [--format json|dot|md]`

### Task 7: E2E test with existing data

**Files:**
- Test with papers in DB (1910.06810, 2405.14831)

**Steps:**
1. Run `ml-research analyze paper` for both papers (if not analyzed)
2. `ml-research graph build diffusion_models --papers 2405.14831,1910.06810`
3. `ml-research graph stats diffusion_models`
4. `ml-research graph query diffusion_models "MATCH (p:Paper) RETURN p.label"`
5. Verify nodes, edges, queries

---

**Schedule:** Tasks 1-5 are core library (~1hr each), Task 6 is CLI integration, Task 7 is validation.
