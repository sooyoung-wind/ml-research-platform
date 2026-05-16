# ML Research Platform

Autonomous ML Research Infrastructure: Paper Discovery → Code Generation Pipeline

## Overview

This platform automates the ML research workflow:

1. **Paper Discovery** — Automatically find relevant ML papers from arXiv, Semantic Scholar, PapersWithCode
2. **Paper Processing** — Download PDFs, parse into structured data (sections, equations, algorithms)
3. **Code Generation** — Generate code implementations from papers using PaperCoder / DeepCode
4. **Orchestration** — Schedule, automate, and report results via Notion/Discord

## Quick Start

```bash
# Setup
cp .env.example .env  # Fill in your API keys
uv sync --extra dev

# Discover papers
ml-research discover --topic "diffusion models" --top 10

# Process a paper
ml-research process --paper-id <ARXIV_ID>

# Generate code from paper
ml-research codegen --paper-id <ARXIV_ID> --engine papercoder
```

## Architecture

```
Paper Discovery → Paper Processing → Code Generation → Validation → GitHub Push
     (Phase 1)        (Phase 2)         (Phase 3)      (Phase 4)
```

## API Keys Required

| API | Purpose | Cost |
|-----|---------|------|
| Semantic Scholar | Paper search + citations | Free (key for higher limits) |
| arXiv | Preprint search | Free |
| OpenAI | Code generation | ~$0.50/paper (o3-mini) |
| CORE | Full-text access | Free (key required) |

## License

MIT
