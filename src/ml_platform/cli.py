"""ML Research Platform — CLI interface.

Provides a Typer-based command-line interface for paper discovery,
processing, code generation, and full pipeline orchestration.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load .env files: project local > ~/.hermes/.env > home .env
for _env_path in [
    Path(".env"),
    Path.home() / ".hermes" / ".env",
    Path.home() / ".env",
]:
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
        break

# Platform defaults from .env (must be after dotenv load)
from ml_platform.config import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER  # noqa: E402


def _check_ollama() -> bool:
    """Check if Ollama server is running locally."""
    import httpx

    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

app = typer.Typer(
    name="ml-research",
    help="ML Research Platform — Paper Discovery & Code Generation Pipeline",
    no_args_is_help=True,
)

discover_app = typer.Typer(help="Paper discovery commands")
app.add_typer(discover_app, name="discover")

process_app = typer.Typer(help="Paper processing commands")
app.add_typer(process_app, name="process")

codegen_app = typer.Typer(help="Code generation commands")
app.add_typer(codegen_app, name="codegen")

run_app = typer.Typer(help="Full pipeline orchestration commands")
app.add_typer(run_app, name="run")

setup_app = typer.Typer(help="Setup and patching commands")
app.add_typer(setup_app, name="setup")

console = Console()


# ── Discover commands ──────────────────────────────────────────────────────


@discover_app.command("search")
def discover_search(
    topic: str = typer.Argument(help="Search topic"),
    top: int = typer.Option(10, "--top", "-n", help="Number of results"),
    source: str = typer.Option("all", "--source", "-s", help="Source: all, arxiv, semantic_scholar, huggingface"),
) -> None:
    """Search for papers on a topic across multiple sources.

    Args:
        topic: The search topic to look for.
        top: Maximum number of results to return.
        source: Paper source to search. One of "all", "arxiv",
            "semantic_scholar", or "huggingface".
    """
    from ml_platform.discovery.pipeline import DiscoveryPipeline

    console.print(f"\n[bold blue]🔍 Searching:[/] {topic} (top {top}, source: {source})\n")

    sources = None if source == "all" else [source]
    pipeline = DiscoveryPipeline()

    with console.status("Fetching papers..."):
        result = asyncio.run(pipeline.search(query=topic, top_n=top, sources=sources))

    # Display results
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=6)
    table.add_column("Title", min_width=40, max_width=60)
    table.add_column("Citations", width=9, justify="right")
    table.add_column("Code", width=5, justify="center")
    table.add_column("Source", width=8)
    table.add_column("Published", width=10)

    for i, paper in enumerate(result.papers, 1):
        title = paper.title[:58] + ".." if len(paper.title) > 58 else paper.title
        citations = str(paper.citation_count) if paper.citation_count else "-"
        code = "✅" if paper.has_code else "—"
        score = f"{paper.composite_score:.2f}" if paper.composite_score else "-"
        source = paper.source.value.replace("_", " ").title()
        published = paper.published_date.strftime("%Y-%m-%d") if paper.published_date else "-"

        table.add_row(str(i), score, title, citations, code, source, published)

    console.print(table)
    console.print(
        f"\n[dim]Found {result.total_found} papers in {result.duration_seconds}s "
        f"(showing top {len(result.papers)})[/dim]"
    )


@discover_app.command("daily")
def discover_daily(
    top: int = typer.Option(10, "--top", "-n", help="Results per topic"),
) -> None:
    """Run daily paper discovery for all configured topics.

    Args:
        top: Maximum number of results per topic.
    """
    from ml_platform.config import config
    from ml_platform.discovery.pipeline import DiscoveryPipeline

    console.print("\n[bold blue]📡 Daily Paper Discovery[/]")
    console.print(f"[dim]Topics: {', '.join(config.DEFAULT_TOPICS)}[/]\n")

    pipeline = DiscoveryPipeline()
    results = asyncio.run(pipeline.daily_discovery(top_n=top))

    total_papers = 0
    for result in results:
        total_papers += result.total_found
        console.print(
            f"  [bold]{result.query}[/]: {result.total_found} found, "
            f"{len(result.papers)} ranked ({result.duration_seconds}s)"
        )

    console.print(f"\n[bold green]Total: {total_papers} papers across {len(results)} topics[/]")


@discover_app.command("trending")
def discover_trending(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of trending papers"),
) -> None:
    """Show today's trending papers from HuggingFace.

    Args:
        limit: Maximum number of trending papers to display.
    """
    from ml_platform.discovery.pipeline import DiscoveryPipeline

    console.print("\n[bold blue]🔥 HuggingFace Trending Papers[/]\n")

    pipeline = DiscoveryPipeline()
    papers = asyncio.run(pipeline.trending(limit=limit))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Upvotes", width=8, justify="right")
    table.add_column("Title", min_width=40, max_width=60)
    table.add_column("Code", width=5, justify="center")
    table.add_column("Stars", width=6, justify="right")

    for i, paper in enumerate(papers, 1):
        title = paper.title[:58] + ".." if len(paper.title) > 58 else paper.title
        code = "✅" if paper.code_url else "—"
        stars = "-"
        table.add_row(str(i), str(paper.upvotes), title, code, stars)

    console.print(table)
    console.print(f"\n[dim]Showing {len(papers)} trending papers from HuggingFace[/dim]")


# ── Process commands ───────────────────────────────────────────────────────


@process_app.command("paper")
def process_paper(
    paper_id: str = typer.Argument(help="Paper arXiv ID (e.g. 2312.00752)"),
    use_grobid: bool = typer.Option(False, "--grobid", help="Use GROBID for structured parsing"),
    no_enrich: bool = typer.Option(False, "--no-enrich", help="Skip metadata enrichment"),
    force: bool = typer.Option(False, "--force", help="Re-process even if already done"),
) -> None:
    """Process a paper: download PDF, extract text, enrich metadata.

    Args:
        paper_id: Paper arXiv ID (e.g. ``2312.00752``).
        use_grobid: Whether to use GROBID for structured parsing.
        no_enrich: Whether to skip metadata enrichment.
        force: Whether to re-process even if already done.
    """
    from ml_platform.models import Paper, PaperSource
    from ml_platform.processing.processor import PaperProcessor

    console.print(f"\n[bold blue]📄 Processing paper:[/] {paper_id}\n")

    paper = Paper(
        paper_id=f"arxiv_{paper_id}",
        arxiv_id=paper_id,
        title=paper_id,
        source=PaperSource.ARXIV,
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )

    processor = PaperProcessor(use_grobid=use_grobid, enrich_metadata=not no_enrich)
    result = asyncio.run(processor.process_paper(paper, force=force))

    if result.success:
        console.print("[bold green]✅ Processing complete![/]")
    else:
        console.print(f"[bold red]❌ Processing failed:[/] {result.error}")

    if result.download.get("success"):
        size_kb = (result.download.get("size_bytes", 0) or 0) / 1024
        console.print(f"  PDF: {result.download.get('path')} ({size_kb:.0f} KB)")

    if result.extracted.get("success"):
        method = result.extracted.get("extraction_method", "?")
        if method == "grobid":
            console.print(
                f"  Parse (GROBID): {result.extracted.get('sections', 0)} sections, "
                f"{result.extracted.get('references', 0)} refs"
            )
        else:
            console.print(
                f"  Text extracted: {result.extracted.get('pages', 0)} pages, "
                f"{result.extracted.get('chars', 0):,} chars (PyPDF2)"
            )

    if result.enriched:
        console.print(
            f"  Enriched: citations={paper.citation_count}, "
            f"code={'yes' if paper.has_code else 'no'}"
        )

    console.print(f"  [dim]Completed in {result.duration:.1f}s[/dim]")


@process_app.command("batch")
def process_batch(
    topic: str = typer.Argument(help="Search topic to discover and process"),
    top: int = typer.Option(5, "--top", "-n", help="Number of papers to process"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Discover papers on a topic and process them all.

    Args:
        topic: Search topic to discover and process.
        top: Number of papers to process.
        force: Whether to force re-processing of already-processed papers.
    """
    from ml_platform.discovery.pipeline import DiscoveryPipeline
    from ml_platform.processing.processor import PaperProcessor

    console.print(f"\n[bold blue]🔄 Batch processing:[/] {topic} (top {top})\n")

    with console.status("[bold]Discovering papers..."):
        discovery = DiscoveryPipeline()
        discover_result = asyncio.run(discovery.search(query=topic, top_n=top))

    console.print(f"  Found {discover_result.total_found} papers, processing top {len(discover_result.papers)}\n")

    processor = PaperProcessor()
    results = asyncio.run(processor.process_batch(discover_result.papers, force=force))

    success = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total_time = sum(r.duration for r in results)

    console.print(f"\n[bold]Results:[/] {success} success, {failed} failed ({total_time:.1f}s total)")
    for r in results:
        icon = "✅" if r.success else "❌"
        console.print(f"  {icon} {r.paper_id} ({r.duration:.1f}s)")


@process_app.command("grobid")
def process_grobid_status() -> None:
    """Check GROBID service status."""
    from ml_platform.processing.grobid_client import GrobidClient

    console.print("\n[bold blue]🔧 GROBID Service Status[/]\n")

    async def check():
        async with GrobidClient() as client:
            return await client.check_health()

    healthy = asyncio.run(check())

    if healthy:
        console.print("[bold green]✅ GROBID is running[/] (http://localhost:8070)")
    else:
        console.print("[bold red]❌ GROBID is not responding[/]")
        console.print("[dim]Start with: docker run -d --name grobid --rm -p 8070:8070 lfoppiano/grobid:0.8.1[/dim]")


# ── Codegen commands ───────────────────────────────────────────────────────


@codegen_app.command("generate")
def codegen_generate(
    paper_id: str = typer.Argument(help="Paper arXiv ID (e.g. 2312.00752)"),
    mode: str = typer.Option("optimized", "--mode", "-m", help="Mode: optimized, comprehensive"),
    provider: str = typer.Option(
        DEFAULT_LLM_PROVIDER, "--provider", "-p",
        help="LLM provider: openai, anthropic, google, ollama",
    ),
    model: str = typer.Option(
        DEFAULT_LLM_MODEL, "--model",
        help="Model name (e.g. gpt-4o, qwen3:8b, glm-5.1:cloud)",
    ),
    output: str = typer.Option("", "--output", "-o", help="Output directory"),
) -> None:
    """Generate code from a paper using DeepCode multi-agent pipeline.

    Args:
        paper_id: Paper arXiv ID (e.g. ``2312.00752``).
        mode: Code generation mode. One of "optimized" or "comprehensive".
        provider:        LLM provider. One of "openai", "anthropic", "google", or "ollama".
        model: Specific model name (e.g. ``gpt-4o``,
            ``claude-sonnet-4-20250514``). Empty string uses the provider
            default.
        output: Output directory for generated files. Empty string uses
            the default ``data/codegen`` directory.
    """
    from ml_platform.codegen.deepcode_runner import DeepCodeConfig, DeepCodeRunner

    console.print(f"\n[bold blue]⚙️ Generating code for:[/] {paper_id}")
    console.print(f"  Mode: {mode} | Provider: {provider}")
    if model:
        console.print(f"  Model: {model}")

    # First, ensure we have the PDF
    from ml_platform.models import Paper, PaperSource
    from ml_platform.processing.pdf_downloader import PDFDownloader

    paper = Paper(
        paper_id=f"arxiv_{paper_id}",
        arxiv_id=paper_id,
        title=paper_id,
        source=PaperSource.ARXIV,
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )

    # Download PDF if not cached
    pdf_path = os.path.join("data", "pdfs", f"{paper_id}.pdf")
    if not os.path.exists(pdf_path):
        console.print("\n[dim]Downloading PDF...[/]")
        async def dl():
            async with PDFDownloader() as d:
                return await d.download_paper(paper)
        dl_result = asyncio.run(dl())
        if not dl_result.success:
            console.print(f"[red]PDF download failed: {dl_result.error}[/]")
            raise typer.Exit(1)
        pdf_path = str(dl_result.path) if dl_result.path else pdf_path

    # Generate code
    dc_config = DeepCodeConfig(
        llm_provider=provider,
        model_name=model or "gpt-4o",
        output_base_dir=output or os.path.join("data", "codegen"),
    )
    runner = DeepCodeRunner(dc_config)

    with console.status("[bold]Running DeepCode pipeline... (this may take several minutes)[/]"):
        result = asyncio.run(runner.generate(
            pdf_path,
            paper_id=paper_id,
            mode=mode,
        ))

    if result.success:
        console.print(f"\n[bold green]✅ Code generation complete![/]")
        console.print(f"  Output: {result.output_dir}")
        console.print(f"  Files: {len(result.files_generated)}")
        for f in result.files_generated[:15]:
            console.print(f"    📄 {f}")
        if len(result.files_generated) > 15:
            console.print(f"    ... and {len(result.files_generated) - 15} more")
        console.print(f"  Duration: {result.duration_seconds:.1f}s")
    else:
        console.print(f"\n[bold red]❌ Code generation failed:[/] {result.error}")


@codegen_app.command("status")
def codegen_status() -> None:
    """Show codegen configuration and available providers."""
    import os

    console.print("\n[bold blue]⚙️ Code Generation Status[/]\n")

    providers = {
        "OpenAI": bool(os.environ.get("OPENAI_API_KEY")),
        "Anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "Google": bool(
            os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        ),
        "Ollama (local)": _check_ollama(),
    }

    for name, configured in providers.items():
        icon = "✅" if configured else "❌"
        console.print(f"  {icon} {name}: {'configured' if configured else 'not configured'}")

    # Check output dir
    output_dir = os.path.join("data", "codegen")
    if os.path.exists(output_dir):
        count = sum(1 for _ in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, _)))
        console.print(f"\n  📁 Output: {output_dir} ({count} projects)")
    else:
        console.print(f"\n  📁 Output: {output_dir} (not yet created)")

    console.print("\n  [dim]Usage: ml-research codegen generate 2312.00752 --mode optimized[/dim]")


# ── Status command ─────────────────────────────────────────────────────────


@app.command("status")
def status() -> None:
    """Show platform status and database stats."""
    from ml_platform.db import PapersDB

    db = PapersDB()
    stats = db.get_stats()

    console.print(Panel("[bold]ML Research Platform Status[/]", style="blue"))
    console.print(f"  Database: [green]{db.db_path}[/]")
    console.print(f"  Total papers: [bold]{stats['total_papers']}[/]")
    console.print(f"  With code:    {stats['with_code']}")
    console.print(f"  Without code: {stats['without_code']}")
    console.print(f"  By status:    {stats['by_status']}")

@app.command("version")
def version() -> None:
    """Show version."""
    from ml_platform import __version__

    console.print(f"[bold]ml-research-platform[/] v{__version__}")

# ── Run commands (orchestration) ──────────────────────────────────────────


@run_app.command("paper")
def run_paper(
    paper_id: str = typer.Argument(help="Paper arXiv ID (e.g. 2312.00752)"),
    mode: str = typer.Option("optimized", "--mode", "-m", help="Codegen mode: optimized, comprehensive"),
    provider: str = typer.Option(
        DEFAULT_LLM_PROVIDER, "--provider", "-p",
        help="LLM provider: openai, anthropic, google, ollama",
    ),
    model: str = typer.Option("", "--model", help="Specific model name"),
    skip_codegen: bool = typer.Option(False, "--skip-codegen", help="Skip code generation"),
    no_push: bool = typer.Option(False, "--no-push", help="Skip GitHub push"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip Notion reporting"),
    force: bool = typer.Option(False, "--force", help="Force re-processing"),
) -> None:
    """Run the full pipeline on a single paper.

    Executes download → process → codegen → push → report.

    Args:
        paper_id: Paper arXiv ID (e.g. ``2312.00752``).
        mode: Code generation mode. One of "optimized" or "comprehensive".
        provider:        LLM provider. One of "openai", "anthropic", "google", or "ollama".
        model: Specific model name. Empty string uses the provider default.
        skip_codegen: Whether to skip the code generation step.
        no_push: Whether to skip pushing to GitHub.
        no_report: Whether to skip reporting to Notion.
        force: Whether to force re-processing of already-processed papers.
    """
    from ml_platform.orchestration.orchestrator import ResearchOrchestrator

    console.print(f"\n[bold blue]🚀 Full pipeline:[/] {paper_id}")
    console.print(f"  Mode: {mode} | Provider: {provider}\n")

    orchestrator = ResearchOrchestrator(
        codegen_mode=mode,
        codegen_provider=provider,
        codegen_model=model,
        skip_codegen=skip_codegen,
        push_to_github=not no_push,
        report_to_notion=not no_report,
        progress_callback=lambda stage, pct, msg: console.print(
            f"  [{stage}] {pct:.0f}% — {msg}"
        ),
    )

    with console.status("[bold]Running pipeline..."):
        result = asyncio.run(orchestrator.run_paper(paper_id, force=force))

    # Display result
    if result.success:
        console.print("\n[bold green]✅ Pipeline complete![/]")
    else:
        console.print(f"\n[bold red]❌ Pipeline failed:[/] {result.error}")

    table = Table(show_header=False, box=None)
    table.add_column("key", style="dim")
    table.add_column("value")
    table.add_row("PDF", result.pdf_path or "—")
    table.add_row("Text", f"{result.text_chars:,} chars" if result.text_chars else "—")
    table.add_row("Enriched", "✅" if result.enriched else "—")
    table.add_row("Code", f"{len(result.code_files)} files" if result.code_files else "—")
    table.add_row("Repo", result.repo_url or "—")
    table.add_row("Notion", result.notion_url or "—")
    table.add_row("Duration", f"{result.total_duration:.1f}s")
    console.print(table)


@run_app.command("topic")
def run_topic(
    topic: str = typer.Argument(help="Search topic"),
    top: int = typer.Option(5, "--top", "-n", help="Number of papers to process"),
    mode: str = typer.Option("optimized", "--mode", "-m", help="Codegen mode"),
    provider: str = typer.Option(
        DEFAULT_LLM_PROVIDER, "--provider", "-p",
        help="LLM provider: openai, anthropic, google, ollama",
    ),
    skip_codegen: bool = typer.Option(False, "--skip-codegen", help="Skip code generation"),
    no_push: bool = typer.Option(False, "--no-push", help="Skip GitHub push"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip Notion reporting"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Discover papers on a topic and run the full pipeline.

    Args:
        topic: Search topic to discover papers for.
        top: Number of papers to process.
        mode: Code generation mode.
        provider: LLM provider.
        skip_codegen: Whether to skip code generation.
        no_push: Whether to skip GitHub push.
        no_report: Whether to skip Notion reporting.
        force: Whether to force re-processing.
    """
    from ml_platform.orchestration.orchestrator import ResearchOrchestrator

    console.print(f"\n[bold blue]🚀 Full pipeline (topic):[/] {topic} (top {top})\n")

    orchestrator = ResearchOrchestrator(
        codegen_mode=mode,
        codegen_provider=provider,
        skip_codegen=skip_codegen,
        push_to_github=not no_push,
        report_to_notion=not no_report,
        max_concurrent=2,
        progress_callback=lambda stage, pct, msg: console.print(
            f"  [{stage}] {pct:.0f}% — {msg}"
        ),
    )

    with console.status("[bold]Running pipeline..."):
        result = asyncio.run(orchestrator.run_topic(topic, top_n=top, force=force))

    console.print(f"\n[bold]Results:[/] {result.successful} ✅  {result.failed} ❌  {result.skipped} ⏭")
    for r in result.results:
        if r.skipped:
            console.print(f"  ⏭ {r.paper_id} (skipped — already has code)")
        elif r.success:
            repo = f" → {r.repo_url}" if r.repo_url else ""
            console.print(f"  ✅ {r.paper_title[:50]}{repo} ({r.total_duration:.1f}s)")
        else:
            console.print(f"  ❌ {r.paper_id}: {r.error[:60]}")
    console.print(f"\n  [dim]Total: {result.total_duration:.1f}s[/dim]")


@run_app.command("daily")
def run_daily(
    top: int = typer.Option(10, "--top", "-n", help="Papers per topic"),
    mode: str = typer.Option("optimized", "--mode", "-m", help="Codegen mode"),
    provider: str = typer.Option(
        DEFAULT_LLM_PROVIDER, "--provider", "-p",
        help="LLM provider: openai, anthropic, google, ollama",
    ),
    skip_codegen: bool = typer.Option(False, "--skip-codegen", help="Skip code generation"),
    no_push: bool = typer.Option(False, "--no-push", help="Skip GitHub push"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip Notion reporting"),
) -> None:
    """Run daily pipeline for all configured topics.

    Args:
        top: Number of papers per topic to process.
        mode: Code generation mode.
        provider: LLM provider.
        skip_codegen: Whether to skip code generation.
        no_push: Whether to skip GitHub push.
        no_report: Whether to skip Notion reporting.
    """
    from ml_platform.config import config
    from ml_platform.orchestration.orchestrator import ResearchOrchestrator

    console.print(f"\n[bold blue]🚀 Daily Pipeline Run[/]")
    console.print(f"  Topics: {', '.join(config.DEFAULT_TOPICS)}")
    console.print(f"  Top {top} per topic | Mode: {mode} | Provider: {provider}\n")

    orchestrator = ResearchOrchestrator(
        codegen_mode=mode,
        codegen_provider=provider,
        skip_codegen=skip_codegen,
        push_to_github=not no_push,
        report_to_notion=not no_report,
        max_concurrent=2,
        progress_callback=lambda stage, pct, msg: console.print(
            f"  [{stage}] {pct:.0f}% — {msg}"
        ),
    )

    with console.status("[bold]Running daily pipeline... (this may take a while)"):
        results = asyncio.run(orchestrator.run_daily(top_n=top))

    total_success = sum(r.successful for r in results)
    total_failed = sum(r.failed for r in results)
    total_skipped = sum(r.skipped for r in results)

    console.print(f"\n[bold green]Daily Run Complete[/]")
    for r in results:
        console.print(
            f"  {r.topic}: {r.successful} ✅  {r.failed} ❌  {r.skipped} ⏭"
            f" ({r.total_duration:.1f}s)"
        )
    console.print(
        f"\n  Total: {total_success} success, {total_failed} failed, {total_skipped} skipped"
    )


# ── Setup commands ─────────────────────────────────────────────────────


@setup_app.command("deepcode")
def setup_deepcode() -> None:
    """Patch DeepCode package (missing modules + Ollama support)."""
    from ml_platform.codegen.deepcode_setup import run_setup

    run_setup()


if __name__ == "__main__":
    app()
