# ML Research Platform

Autonomous ML Research Infrastructure: from paper discovery to knowledge compilation.

[![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/sooyoung-wind/ml-research-platform/releases/tag/v0.2.0)
[![Python](https://img.shields.io/badge/python-3.11+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

## Overview

ML Research Platform automates the full ML research lifecycle in a single CLI:

```
Paper Discovery → 5W1H Analysis → Knowledge Graph → Research Map → Trend Analysis → LLM Wiki
     (Phase 1)       (Phase 5)       (Phase 6)       (Phase 8)      (Phase 7)       (Phase 9)
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Paper Discovery** | arXiv search with async httpx client |
| **Paper Processing** | PDF download + text extraction (PyPDF2) |
| **5W1H Analysis** | Who/What/When/Where/Why/How extraction via Ollama LLM |
| **Knowledge Graph** | graphqlite (SQLite + Cypher) with topic-based independent DBs |
| **Research Map** | Interactive pyvis visualization with Label Propagation community detection |
| **Trend Analyzer** | LLM-driven interactive interview + topic-filtered reports |
| **LLM Wiki** | OpenKB integration — auto-compile papers into searchable knowledge base |
| **Code Generation** | DeepCode workflow for paper-to-code implementation |
| **E2E Pipeline** | Single command orchestrating all stages |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/sooyoung-wind/ml-research-platform.git
cd ml-research-platform
cp .env.example .env        # Configure LLM provider
uv sync --extra dev

# Check status
ml-research status

# Run full E2E pipeline (dry run first)
ml-research run e2e "diffusion models" --dry-run
ml-research run e2e "RAG" --top 10
```

## CLI Commands (30+)

### Discovery & Processing

```bash
ml-research discover search --topic "knowledge graphs" --top 10
ml-research process download --paper-id 2312.00752
ml-research process parse    --paper-id 2312.00752
```

### Analysis

```bash
ml-research analyze paper 2312.00752              # 5W1H + evidence extraction
ml-research trend                                # Statistical trend analysis
ml-research trend-interview                      # LLM-driven interactive interview
```

### Knowledge Graph

```bash
ml-research graph-build --topic "ml_research"     # Build from paper analyses
ml-research graph-query --topic "ml_research" \
    --query "MATCH (p:Paper)-[:USES_METHOD]->(m) RETURN p,m"
ml-research graph-stats --topic "ml_research"     # Node/edge counts
ml-research graph-list                            # List all topic graphs
ml-research graph-view --topic "ml_research"      # Interactive HTML visualization
ml-research graph-export --topic "ml_research"    # Export as JSON
```

### Research Map

```bash
ml-research map --topic "ml_research"             # From knowledge graph
ml-research map --source papers                   # From paper database
```

Generates interactive HTML with:
- Label Propagation community detection for topic clustering
- Clickable detail side-panel (abstract, authors, citations)
- Tailwind CSS dark theme with type-based icons

### LLM Wiki (OpenKB)

```bash
ml-research wiki-init                             # Initialize with Ollama backend
ml-research wiki-import --all                     # Import all papers from DB
ml-research wiki-query "What is RAG?"             # Keyword search + LLM synthesis
ml-research wiki-list                             # List indexed documents
ml-research wiki-status                           # Wiki statistics
ml-research wiki-lint                             # Structure/semantic linting
```

### E2E Pipeline

```bash
ml-research run e2e "transformer attention" --top 5     # Full pipeline
ml-research run e2e "RAG" --dry-run                     # Plan only
ml-research run e2e "KG" --skip-analysis --skip-wiki    # Selective stages
ml-research run e2e "GNN" --force                       # Re-process everything
```

Pipeline stages (each skippable with `--skip-*`):

| Stage | Flag | Description |
|-------|------|-------------|
| 1. Discovery | `--skip-discovery` | arXiv keyword search |
| 2. Analysis | `--skip-analysis` | 5W1H + evidence extraction |
| 3. Knowledge Graph | `--skip-graph` | graphqlite KG build |
| 4. Research Map | `--skip-map` | pyvis HTML visualization |
| 5. Trend Analysis | `--skip-trend` | Statistical trend report |
| 6. Wiki Import | `--skip-wiki` | OpenKB compilation |

## Architecture

```
src/ml_platform/
├── analysis/
│   ├── analyzer.py           # 5W1H analysis via Ollama
│   ├── research_map.py       # pyvis + Tailwind research map
│   ├── trends.py             # Statistical trend analyzer
│   └── interview.py          # LangGraph LLM interview
├── config.py                 # .env-based configuration
├── db.py                     # SQLite paper database
├── discovery/
│   └── arxiv_client.py       # Async arXiv search
├── graph/
│   └── knowledge_graph.py    # graphqlite KG builder
├── models.py                 # Pydantic data models
├── orchestration/
│   └── orchestrator.py       # Full pipeline orchestrator
├── processing/
│   └── processor.py          # PDF download + text extraction
├── wiki/
│   └── __init__.py           # OpenKB WikiManager
├── cli.py                    # Typer CLI (30+ commands)
└── __init__.py               # v0.2.0
```

## Configuration

All settings are controlled via `.env`:

```bash
# LLM Configuration
ML_DEFAULT_LLM_PROVIDER=ollama
ML_DEFAULT_LLM_MODEL=gemma4:31b-cloud
OLLAMA_BASE_URL=http://localhost:11434

# Optional (for cloud providers)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

No hardcoded model names — everything reads from environment variables.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Package Manager | uv |
| CLI Framework | Typer + Rich |
| Database | SQLite (papers.db) |
| Knowledge Graph | graphqlite (Cypher on SQLite) |
| Graph Visualization | pyvis |
| LLM Backend | LiteLLM + Ollama |
| Wiki Engine | OpenKB (VectifyAI) |
| Interview Agent | LangGraph 1.2 (HITL) |
| Code Generation | DeepCode |

## Data

```
data/
├── papers.db          # 20 papers (arXiv, 2020-2026)
├── graphs/            # Topic KGs (graph_{topic}.db)
├── maps/              # Interactive HTML maps
├── analyses/          # 5W1H analysis JSON files
├── wiki/              # OpenKB wiki data
└── trend_report.md    # Latest trend report
```

## Project Stats

- **42 Python modules** — 15,000+ lines of code
- **30+ CLI commands** across 7 groups
- **20 papers** indexed from arXiv
- **3 knowledge graphs** (knowledge_graph, ml_research, knowledge_reasoning)
- **19 wiki summaries** + **35 concepts** auto-generated
- **E2E pipeline** tested end-to-end

## License

MIT
