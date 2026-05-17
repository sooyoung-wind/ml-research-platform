"""ML Research Platform — Entity extraction from paper analyses.

Extracts structured entities (Authors, Methods, Datasets, Concepts,
Institutions, Venues) from PaperAnalysis results and links them
with typed edges.
"""

from __future__ import annotations

import re
from typing import Any

from ml_platform.analysis.models import PaperAnalysis
from ml_platform.graph.models import EdgeType, GraphEdge, GraphNode, NodeType


def extract_entities(
    analysis: PaperAnalysis,
    paper_id: str | None = None,
) -> list[GraphNode]:
    """Extract graph nodes from a paper analysis.

    Extracts Author, Method, Dataset, Concept, Institution, and Venue
    nodes based on the structured 5W1H analysis fields.

    Args:
        analysis: The completed paper analysis.
        paper_id: Optional paper ID override.

    Returns:
        List of GraphNode objects (Paper node is always first).
    """
    pid = paper_id or analysis.paper_id
    nodes: list[GraphNode] = []

    # Paper node (always present)
    nodes.append(GraphNode(
        node_id=_paper_node_id(pid),
        node_type=NodeType.PAPER,
        label=_truncate(analysis.summary or pid, 120),
        properties={
            "paper_id": pid,
            "year": analysis.five_w1h.when[:4] if analysis.five_w1h.when else 0,
            "domain": analysis.domain,
            "methodology_type": analysis.methodology_type,
        },
    ))

    # Authors from "who"
    _extract_authors(analysis.five_w1h.who, nodes)

    # Methods from "how"
    _extract_methods(analysis.five_w1h.how, nodes)

    # Datasets from "how" and "what"
    _extract_datasets(analysis.five_w1h.how, analysis.five_w1h.what, nodes)

    # Concepts from domain + key contributions
    _extract_concepts(analysis.domain, analysis.key_contributions, nodes)

    # Institutions from "who"
    _extract_institutions(analysis.five_w1h.who, nodes)

    # Venue from "where"
    venue = analysis.five_w1h.where.strip()
    if venue and not venue.startswith("Not specified"):
        nodes.append(GraphNode(
            node_id=f"venue_{_slugify(venue)}",
            node_type=NodeType.VENUE,
            label=venue,
        ))

    return nodes


def extract_edges(
    analysis: PaperAnalysis,
    nodes: list[GraphNode],
    references: list[dict[str, Any]] | None = None,
) -> list[GraphEdge]:
    """Extract graph edges from analysis and node relationships.

    Args:
        analysis: The paper analysis.
        nodes: Previously extracted nodes for this paper.
        references: Optional list of reference dicts with paper_id keys.

    Returns:
        List of GraphEdge objects.
    """
    edges: list[GraphEdge] = []
    pid = analysis.paper_id
    paper_nid = _paper_node_id(pid)

    node_map = {n.node_id: n for n in nodes}

    # Author -> CONTRIBUTES_TO -> Paper
    for node in nodes:
        if node.node_type == NodeType.AUTHOR:
            edges.append(GraphEdge(
                source_id=node.node_id,
                target_id=paper_nid,
                edge_type=EdgeType.CONTRIBUTES_TO,
            ))

    # Paper -> PROPOSES -> Method
    for node in nodes:
        if node.node_type == NodeType.METHOD:
            edges.append(GraphEdge(
                source_id=paper_nid,
                target_id=node.node_id,
                edge_type=EdgeType.PROPOSES,
            ))

    # Paper -> USES -> Method
    for node in nodes:
        if node.node_type == NodeType.METHOD:
            edges.append(GraphEdge(
                source_id=paper_nid,
                target_id=node.node_id,
                edge_type=EdgeType.USES,
            ))

    # Paper -> EVALUATES_ON -> Dataset
    for node in nodes:
        if node.node_type == NodeType.DATASET:
            edges.append(GraphEdge(
                source_id=paper_nid,
                target_id=node.node_id,
                edge_type=EdgeType.EVALUATES_ON,
            ))

    # Paper -> BELONGS_TO -> Venue
    for node in nodes:
        if node.node_type == NodeType.VENUE:
            edges.append(GraphEdge(
                source_id=paper_nid,
                target_id=node.node_id,
                edge_type=EdgeType.BELONGS_TO,
            ))

    # Author -> AFFILIATED_WITH -> Institution
    authors = [n for n in nodes if n.node_type == NodeType.AUTHOR]
    institutions = [n for n in nodes if n.node_type == NodeType.INSTITUTION]
    for author in authors:
        for inst in institutions:
            edges.append(GraphEdge(
                source_id=author.node_id,
                target_id=inst.node_id,
                edge_type=EdgeType.AFFILIATED_WITH,
            ))

    # Paper -> CITES -> Paper (from references)
    if references:
        for ref in references:
            ref_id = ref.get("paper_id") or ref.get("arxiv_id")
            if ref_id:
                edges.append(GraphEdge(
                    source_id=paper_nid,
                    target_id=_paper_node_id(ref_id),
                    edge_type=EdgeType.CITES,
                    properties={"year": ref.get("year", 0)},
                ))

    # Paper -> RELATED_TO -> Concept
    for node in nodes:
        if node.node_type == NodeType.CONCEPT:
            edges.append(GraphEdge(
                source_id=paper_nid,
                target_id=node.node_id,
                edge_type=EdgeType.RELATED_TO,
            ))

    return edges


# ── Internal extraction helpers ──────────────────────────────────────


def _extract_authors(who_text: str, nodes: list[GraphNode]) -> None:
    """Extract author names from the 'who' field.

    Handles formats like:
    - "John Doe, Jane Smith"
    - "John Doe (MIT), Jane Smith (Stanford)"
    - "John Doe and Jane Smith"

    Args:
        who_text: The 'who' field from 5W1H analysis.
        nodes: List to append author nodes to.
    """
    if not who_text or who_text.startswith("Not specified"):
        return

    # Remove parenthetical affiliations for name extraction
    clean = re.sub(r'\([^)]*\)', '', who_text)

    # Split on commas, "and", semicolons
    parts = re.split(r'[,;&]|\band\b', clean)

    seen: set[str] = set()
    for part in parts:
        name = part.strip()
        if not name or len(name) < 3 or len(name) > 80:
            continue
        # Skip if looks like an institution
        inst_keywords = ['university', 'institute', 'lab', 'research', 'college',
                         'department', 'school', 'center', 'team', 'group', 'corp']
        if any(kw in name.lower() for kw in inst_keywords):
            continue
        # Skip if it's a role description
        if any(kw in name.lower() for kw in ['et al', 'researchers', 'authors']):
            continue

        slug = _slugify(name)
        if slug not in seen:
            seen.add(slug)
            nodes.append(GraphNode(
                node_id=f"author_{slug}",
                node_type=NodeType.AUTHOR,
                label=name,
            ))


def _extract_methods(how_text: str, nodes: list[GraphNode]) -> None:
    """Extract method names from the 'how' field.

    Args:
        how_text: The 'how' field from 5W1H analysis.
        nodes: List to append method nodes to.
    """
    if not how_text or how_text.startswith("Not specified"):
        return

    # Common ML/AI method patterns
    method_patterns = [
        r'(?i)\b(PERSONALIZED\s+PAGERANK|PPR)\b',
        r'(?i)\b(PAGE\s*RANK)\b',
        r'(?i)\b(KNOWLEDGE\s+GRAPH)\b',
        r'(?i)\b(RETRIEVAL[- ]AUGMENTED\s+GENERATION|RAG)\b',
        r'(?i)\b(CONVOLUTIONAL\s+(?:NEURAL\s+)?NET|CNN)\b',
        r'(?i)\b(RECURRENT\s+(?:NEURAL\s+)?NET|RNN|LSTM|GRU)\b',
        r'(?i)\b(TRANSFORMER)\b',
        r'(?i)\b(AUTOENCODER)\b',
        r'(?i)\b(GAN|GENERATIVE\s+ADVERSARIAL)\b',
        r'(?i)\b(DIFFUSION\s+MODEL)\b',
        r'(?i)\b(FLOW\s+MATCHING)\b',
        r'(?i)\b(GRPO|GROUP\s+RELATIVE\s+POLICY\s+OPTIMIZATION)\b',
        r'(?i)\b(DPO|DIRECT\s+PREFERENCE\s+OPTIMIZATION)\b',
        r'(?i)\b(LORA|LOW[- ]RANK\s+ADAPTATION)\b',
        r'(?i)\b(PERSONALIZED\s+PAGERANK)\b',
        r'(?i)\b(NAMED\s+ENTITY\s+RECOGNITION|NER)\b',
        r'(?i)\b(OPEN\s+INFORMATION\s+EXTRACTION|OPENIE)\b',
        r'(?i)\b(COLBERT)\b',
        r'(?i)\b(CONTRIEVER)\b',
        r'(?i)\b(EMBEDDING)\b',
        r'(?i)\b(SEMANTIC\s+SEARCH)\b',
        r'(?i)\b(NEURAL\s+NETWORK)\b',
        r'(?i)\b(LES|LARGE\s+EDDY\s+SIMULATION)\b',
        r'(?i)\b(CONVLSTM|CONVOLUTIONAL\s+LSTM)\b',
    ]

    seen: set[str] = set()
    for pattern in method_patterns:
        matches = re.finditer(pattern, how_text)
        for match in matches:
            name = match.group(1).strip()
            # Use the short form if it's a parenthetical
            if any(c.islower() for c in name):
                name = name.upper()
            slug = _slugify(name)
            if slug not in seen:
                seen.add(slug)
                nodes.append(GraphNode(
                    node_id=f"method_{slug}",
                    node_type=NodeType.METHOD,
                    label=name,
                ))


def _extract_datasets(how_text: str, what_text: str, nodes: list[GraphNode]) -> None:
    """Extract dataset names from 'how' and 'what' fields.

    Args:
        how_text: The 'how' field from 5W1H analysis.
        what_text: The 'what' field from 5W1H analysis.
        nodes: List to append dataset nodes to.
    """
    combined = f"{how_text} {what_text}"

    dataset_patterns = [
        r'(?i)\b(2WikiMultiHopQA)\b',
        r'(?i)\b(MultiHopRAG)\b',
        r'(?i)\b(HotpotQA)\b',
        r'(?i)\b(MuSiQue)\b',
        r'(?i)\b(Bamboogle)\b',
        r'(?i)\b(WikiData)\b',
        r'(?i)\b(ImageNet)\b',
        r'(?i)\b(COCO)\b',
        r'(?i)\b(SQuAD)\b',
        r'(?i)\b(GLUE)\b',
        r'(?i)\b(SuperGLUE)\b',
        r'(?i)\b(MMLU)\b',
        r'(?i)\b(GSM8K)\b',
        r'(?i)\b(HumanEval)\b',
        r'(?i)\b(ImageNet)\b',
        r'(?i)\b(CIFAR[- ]?\d+)\b',
        r'(?i)\b(SST[- ]?2)\b',
    ]

    seen: set[str] = set()
    for pattern in dataset_patterns:
        matches = re.finditer(pattern, combined)
        for match in matches:
            name = match.group(1).strip()
            slug = _slugify(name)
            if slug not in seen:
                seen.add(slug)
                nodes.append(GraphNode(
                    node_id=f"dataset_{slug}",
                    node_type=NodeType.DATASET,
                    label=name,
                ))


def _extract_concepts(
    domain: str,
    contributions: list[str],
    nodes: list[GraphNode],
) -> None:
    """Extract concept nodes from domain and key contributions.

    Args:
        domain: Research domain string.
        contributions: List of key contribution strings.
        nodes: List to append concept nodes to.
    """
    seen: set[str] = set()

    # Domain as a concept
    if domain and not domain.startswith("Not specified"):
        slug = _slugify(domain)
        if slug not in seen:
            seen.add(slug)
            nodes.append(GraphNode(
                node_id=f"concept_{slug}",
                node_type=NodeType.CONCEPT,
                label=domain,
            ))

    # Extract key phrases from contributions
    concept_keywords = [
        r'(?i)(long[- ]term\s+memory)',
        r'(?i)(multi[- ]hop\s+reasoning)',
        r'(?i)(information\s+retrieval)',
        r'(?i)(knowledge\s+graph)',
        r'(?i)(turbulence\s+generation)',
        r'(?i)(inflow\s+generation)',
        r'(?i)(graph[- ]based\s+retrieval)',
        r'(?i)(dense\s+retrieval)',
        r'(?i)(sparse\s+retrieval)',
        r'(?i)(entity\s+linking)',
        r'(?i)(passage\s+retrieval)',
    ]

    combined = " ".join(contributions)
    for pattern in concept_keywords:
        matches = re.finditer(pattern, combined)
        for match in matches:
            name = match.group(1).strip().title()
            slug = _slugify(name)
            if slug not in seen:
                seen.add(slug)
                nodes.append(GraphNode(
                    node_id=f"concept_{slug}",
                    node_type=NodeType.CONCEPT,
                    label=name,
                ))


def _extract_institutions(who_text: str, nodes: list[GraphNode]) -> None:
    """Extract institution names from the 'who' field.

    Args:
        who_text: The 'who' field from 5W1H analysis.
        nodes: List to append institution nodes to.
    """
    if not who_text or who_text.startswith("Not specified"):
        return

    inst_keywords = [
        'university', 'institute', 'laboratory', 'lab',
        'research', 'college', 'department', 'school',
        'center', 'institut', 'faculty',
    ]

    # Extract parenthetical affiliations
    parens = re.findall(r'\(([^)]+)\)', who_text)
    seen: set[str] = set()

    for paren in parens:
        paren = paren.strip()
        if any(kw in paren.lower() for kw in inst_keywords):
            slug = _slugify(paren)
            if slug not in seen:
                seen.add(slug)
                nodes.append(GraphNode(
                    node_id=f"inst_{slug}",
                    node_type=NodeType.INSTITUTION,
                    label=paren,
                ))


def _paper_node_id(paper_id: str) -> str:
    """Convert a paper ID to a graph node ID.

    Args:
        paper_id: Paper identifier.

    Returns:
        Node ID like ``paper_2405_14831``.
    """
    return f"paper_{paper_id.replace('.', '_').replace('/', '_')}"


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


def _truncate(text: str, max_len: int = 120) -> str:
    """Truncate text to max_len, adding ellipsis if needed.

    Args:
        text: Input text.
        max_len: Maximum length.

    Returns:
        Truncated text.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
