"""ML Research Platform — Wiki integration module.

Wraps OpenKB (VectifyAI) as the LLM wiki engine, providing:
- KB initialization with Ollama backend
- Paper import pipeline (papers.db → raw/ → wiki compilation)
- CLI subcommands (wiki init/add/query/chat/lint/list/status)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

from ml_platform.config import AppConfig


class WikiManager:
    """Manages the OpenKB-backed research wiki.

    Directory structure:
        data/wiki/
        ├── .openkb/         OpenKB state (config, hashes)
        ├── raw/             Source documents (PDFs, Markdown)
        ├── wiki/
        │   ├── AGENTS.md    Wiki schema (LLM instructions)
        │   ├── index.md     Page catalog
        │   ├── log.md       Operation log
        │   ├── sources/     Converted markdown
        │   ├── summaries/   Per-document summaries
        │   ├── concepts/    Cross-document synthesis
        │   ├── explorations/ Saved queries
        │   └── reports/     Lint reports
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or AppConfig.DATA_DIR
        self.wiki_dir = self.data_dir / "wiki"

    @property
    def kb_dir(self) -> Path:
        return self.wiki_dir

    @property
    def raw_dir(self) -> Path:
        return self.wiki_dir / "raw"

    @property
    def openkb_dir(self) -> Path:
        return self.wiki_dir / ".openkb"

    @property
    def is_initialized(self) -> bool:
        return (self.openkb_dir / "config.yaml").exists()

    # ── Initialization ────────────────────────────────────────────────

    def init(
        self,
        model: str = "",
        language: str = "en",
        api_key: str = "",
    ) -> dict:
        """Initialize the OpenKB knowledge base.

        Args:
            model: LiteLLM model string (e.g. "ollama/gemma4:31b-cloud").
            language: Wiki output language code.
            api_key: API key for the LLM provider.

        Returns:
            Dict with status and paths created.
        """
        from openkb.config import save_config, register_kb
        from openkb.schema import AGENTS_MD

        if self.is_initialized:
            return {"status": "already_exists", "path": str(self.wiki_dir)}

        if not model:
            # Use default from .env
            model = os.environ.get(
                "ML_DEFAULT_LLM_MODEL", "gemma4:31b-cloud"
            )
            # Prepend ollama/ if no provider prefix
            if "/" not in model:
                model = f"ollama/{model}"

        # Create directory structure
        dirs = [
            self.raw_dir,
            self.wiki_dir / "wiki" / "sources" / "images",
            self.wiki_dir / "wiki" / "summaries",
            self.wiki_dir / "wiki" / "concepts",
            self.wiki_dir / "wiki" / "explorations",
            self.wiki_dir / "wiki" / "reports",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Write wiki files
        wiki_sub = self.wiki_dir / "wiki"
        (wiki_sub / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
        (wiki_sub / "index.md").write_text(
            "# ML Research Knowledge Base\n\n"
            "> Auto-compiled from discovered papers.\n\n"
            "## Documents\n\n## Concepts\n\n## Explorations\n",
            encoding="utf-8",
        )
        (wiki_sub / "log.md").write_text(
            "# Operations Log\n\n", encoding="utf-8"
        )

        # Create .openkb/ state
        self.openkb_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "model": model,
            "language": language,
            "pageindex_threshold": 20,
        }
        save_config(self.openkb_dir / "config.yaml", config)
        (self.openkb_dir / "hashes.json").write_text(
            json.dumps({}), encoding="utf-8"
        )

        # Write .env with API key or Ollama base URL
        env_path = self.wiki_dir / ".env"
        env_lines = []
        if api_key:
            env_lines.append(f"LLM_API_KEY={api_key}")
        else:
            # For Ollama, we need a dummy key and the base URL
            ollama_url = os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )
            env_lines.append("LLM_API_KEY=ollama")
            env_lines.append(
                f"OLLAMA_API_BASE={ollama_url}"
            )
        env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

        # Register in global config
        register_kb(self.wiki_dir)

        return {
            "status": "created",
            "path": str(self.wiki_dir),
            "model": model,
            "language": language,
        }

    # ── Paper import ──────────────────────────────────────────────────

    def import_paper(self, paper_id: str) -> dict:
        """Import a paper from papers.db into the wiki.

        Downloads the PDF (if available) or creates a markdown source
        from the paper metadata + abstract, then compiles into wiki.

        Returns:
            Dict with status, doc_name, and files created.
        """
        from openkb.cli import add_single_file

        if not self.is_initialized:
            return {"status": "error", "message": "Wiki not initialized"}

        from ml_platform.db import PapersDB, PaperSource
        db = PapersDB()

        # Try arXiv ID first, then generic
        paper = db.get_paper_by_arxiv(paper_id)
        if not paper:
            paper = db.get_paper(paper_id, PaperSource.ARXIV)
        if not paper:
            return {"status": "error", "message": f"Paper not found: {paper_id}"}

        # Create a markdown source from paper metadata
        doc_name = paper.arxiv_id or paper.paper_id
        safe_name = doc_name.replace("/", "_").replace(":", "_")

        # Check if already imported
        raw_path = self.raw_dir / f"{safe_name}.md"
        if raw_path.exists():
            return {"status": "already_imported", "doc_name": safe_name}

        # Build markdown content
        lines = [
            f"# {paper.title or doc_name}",
            "",
        ]

        if paper.authors:
            author_names = []
            for a in paper.authors:
                if hasattr(a, "name"):
                    author_names.append(a.name)
                else:
                    author_names.append(str(a))
            lines.append(f"**Authors:** {', '.join(author_names)}")
            lines.append("")

        if paper.year:
            lines.append(f"**Year:** {paper.year}")
            lines.append("")

        if paper.categories:
            lines.append(f"**Categories:** {', '.join(paper.categories)}")
            lines.append("")

        if paper.url:
            lines.append(f"**URL:** {paper.url}")
            lines.append("")

        if paper.abstract:
            lines.append("## Abstract")
            lines.append("")
            lines.append(paper.abstract)
            lines.append("")

        if paper.keywords:
            lines.append(f"**Keywords:** {', '.join(paper.keywords)}")
            lines.append("")

        if paper.citation_count:
            lines.append(f"**Citations:** {paper.citation_count}")
            lines.append("")

        raw_path.write_text("\n".join(lines), encoding="utf-8")

        # Compile via OpenKB
        try:
            add_single_file(raw_path, self.wiki_dir)
        except Exception as e:
            return {"status": "compile_error", "doc_name": safe_name, "error": str(e)}

        return {
            "status": "imported",
            "doc_name": safe_name,
            "raw_path": str(raw_path),
        }

    def import_all_papers(self, limit: int = 0) -> dict:
        """Import all papers from papers.db into the wiki.

        Returns:
            Dict with counts: total, imported, skipped, errors.
        """
        from ml_platform.db import PapersDB

        if not self.is_initialized:
            return {"status": "error", "message": "Wiki not initialized"}

        db = PapersDB()
        actual_limit = limit if limit > 0 else 1000
        papers = db.get_papers(limit=actual_limit)

        results = {"total": len(papers), "imported": 0, "skipped": 0, "errors": 0}
        for paper in papers:
            pid = paper.arxiv_id or paper.paper_id
            result = self.import_paper(pid)
            if result["status"] == "imported":
                results["imported"] += 1
            elif result["status"] == "already_imported":
                results["skipped"] += 1
            else:
                results["errors"] += 1

        return results

    # ── Query / Chat / Lint (delegate to OpenKB) ─────────────────────

    def query(self, question: str, save: bool = False) -> str:
        """Query the knowledge base using direct file search + LLM synthesis.

        Falls back to direct wiki search when the OpenKB agent's tool calling
        isn't fully supported by the current LLM provider (e.g. Ollama).
        """
        import litellm
        litellm.drop_params = True

        from openkb.config import load_config, DEFAULT_CONFIG
        from openkb.cli import _setup_llm_key

        config = load_config(self.openkb_dir / "config.yaml")
        _setup_llm_key(self.wiki_dir)
        model = config.get("model", DEFAULT_CONFIG["model"])

        # Collect relevant wiki content
        wiki_content = self._collect_relevant_content(question)

        # Use LLM directly to synthesize answer
        prompt = (
            "You are a research wiki assistant. Answer the question based on "
            "the following wiki content. Cite sources using [[wikilinks]]. "
            "If you can't answer from the content, say so.\n\n"
            f"## Wiki Content\n\n{wiki_content}\n\n"
            f"## Question\n\n{question}\n\n"
            "## Answer\n"
        )

        try:
            import litellm as _litellm
            response = _litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
                drop_params=True,
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            return f"[ERROR] LLM query failed: {e}"

        from openkb.log import append_log
        append_log(self.wiki_dir / "wiki", "query", question)

        if save and answer:
            import re
            slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60]
            explore_dir = self.wiki_dir / "wiki" / "explorations"
            explore_dir.mkdir(parents=True, exist_ok=True)
            (explore_dir / f"{slug}.md").write_text(
                f'---\nquery: "{question}"\n---\n\n{answer}\n',
                encoding="utf-8",
            )

        return answer

    def _collect_relevant_content(self, question: str, max_chars: int = 8000) -> str:
        """Collect relevant wiki content for a query using keyword matching."""
        import re
        words = set(re.findall(r"\w{3,}", question.lower()))

        parts = []

        # 1. Always include index
        index_path = self.wiki_dir / "wiki" / "index.md"
        if index_path.exists():
            parts.append("### Index\n" + index_path.read_text(encoding="utf-8")[:2000])

        # 2. Search summaries for keyword matches
        summaries_dir = self.wiki_dir / "wiki" / "summaries"
        if summaries_dir.exists():
            for md in sorted(summaries_dir.glob("*.md")):
                content = md.read_text(encoding="utf-8")
                # Strip frontmatter for matching
                body = content.split("---", 2)[-1] if content.startswith("---") else content
                content_words = set(re.findall(r"\w{3,}", body.lower()))
                overlap = len(words & content_words)
                if overlap >= 2:
                    parts.append(f"### Summary: {md.stem}\n{content[:1500]}")

        # 3. Search concepts
        concepts_dir = self.wiki_dir / "wiki" / "concepts"
        if concepts_dir.exists():
            for md in sorted(concepts_dir.glob("*.md")):
                content = md.read_text(encoding="utf-8")
                body = content.split("---", 2)[-1] if content.startswith("---") else content
                content_words = set(re.findall(r"\w{3,}", body.lower()))
                overlap = len(words & content_words)
                if overlap >= 2:
                    parts.append(f"### Concept: {md.stem}\n{content[:1200]}")

        combined = "\n\n".join(parts)
        return combined[:max_chars]

    def list_docs(self) -> list[dict]:
        """List all indexed documents."""
        hashes_file = self.openkb_dir / "hashes.json"
        if not hashes_file.exists():
            return []
        hashes = json.loads(hashes_file.read_text(encoding="utf-8"))
        return [
            {
                "name": meta.get("name", "unknown"),
                "type": meta.get("type", "unknown"),
                "doc_name": meta.get("doc_name", ""),
            }
            for meta in hashes.values()
        ]

    def get_status(self) -> dict:
        """Get wiki status."""
        if not self.is_initialized:
            return {"initialized": False}

        wiki_sub = self.wiki_dir / "wiki"
        counts = {}
        for subdir in ["sources", "summaries", "concepts", "explorations", "reports"]:
            path = wiki_sub / subdir
            counts[subdir] = len(list(path.glob("*.md"))) if path.exists() else 0

        raw_count = (
            len([f for f in self.raw_dir.iterdir() if f.is_file()])
            if self.raw_dir.exists()
            else 0
        )

        hashes_file = self.openkb_dir / "hashes.json"
        indexed = 0
        if hashes_file.exists():
            indexed = len(json.loads(hashes_file.read_text(encoding="utf-8")))

        # Read model from config
        from openkb.config import load_config
        config = load_config(self.openkb_dir / "config.yaml")

        return {
            "initialized": True,
            "path": str(self.wiki_dir),
            "model": config.get("model", ""),
            "language": config.get("language", "en"),
            "indexed": indexed,
            "raw_files": raw_count,
            **counts,
        }
