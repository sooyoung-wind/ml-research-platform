# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2025-05-17

### Added

**5W1H Analysis & Evidence Extraction**
- Paper analyzer with 5W1H (Who/What/When/Where/Why/How) extraction via Ollama LLM
- Self-correction verification pass on analysis results
- Evidence sentence mapping with confidence scoring
- Reference chain extraction from parsed papers

**Knowledge Graph (graphqlite)**
- Knowledge graph builder with topic-based independent SQLite DBs
- Node types: Paper, Author, Category, Method, Dataset
- Edge types: CITES, USES_METHOD, BELONGS_TO, AUTHORED_BY
- `graph-build`, `graph-query`, `graph-stats`, `graph-list`, `graph-export`, `graph-view` CLI commands
- Interactive pyvis HTML visualization with dark theme

**Research Map**
- Network-based research landscape visualization
- Label Propagation community detection for topic clustering
- Pyvis interactive graph with type-based node sizing and coloring
- Clickable detail side-panel with Tailwind CSS dark theme
- SVG icons and color badges per node type
- `map` CLI command with `--source graph|papers` option

**Trend Analyzer**
- Statistical trend analysis across paper metadata
- LLM-driven interactive interview mode (`trend-interview`)
- Dynamic question generation based on user context
- Topic-filtered trend reports with research gap identification
- Markdown report export

**LLM Wiki (OpenKB Integration)**
- WikiManager wrapping VectifyAI/OpenKB as wiki engine
- Automatic paper-to-wiki import pipeline (papers.db → markdown → LLM summary + concepts)
- Direct keyword search + LLM synthesis for wiki queries
- LiteLLM + Ollama (gemma4:31b-cloud) backend
- 7 CLI commands: `wiki-init`, `wiki-add`, `wiki-import`, `wiki-query`, `wiki-list`, `wiki-status`, `wiki-lint`
- 20 papers imported → 19 summaries + 35 concepts auto-generated

**E2E Pipeline**
- Single-command `run e2e` orchestrating all 6 stages
- Granular `--skip-*` flags for each stage
- `--dry-run` mode for pipeline planning
- `--force` for re-processing existing results

### Changed
- CLI description updated to reflect full platform scope
- `status` command enhanced with KG, Analyses, Maps, Wiki stats
- All LLM defaults externalized to `.env` (no hardcoded models)

### Fixed
- Duplicate headings in research map HTML (pyvis `<h1>` + Tailwind `<h2>`)
- KnowledgeGraph constructor → `KnowledgeGraph.open()` class method
- graphqlite `properties()` JSON string handling in Cypher queries
- Files:0 bug in code generation workflow (absolute path fix)
- OpenAI client base URL configuration via secrets.yaml

## [0.1.0] — 2025-05-15

### Added
- Paper discovery via arXiv search API
- Paper processing: download PDF → extract text (PyPDF2)
- Code generation: DeepCode workflow with Ollama LLM
- Full pipeline orchestration: `run paper`, `run topic`, `run daily`
- SQLite paper database with entity resolution
- CLI with 20+ commands
- E2E test suite (5/5 passing)
