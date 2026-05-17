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

analyze_app = typer.Typer(help="Paper analysis commands")
app.add_typer(analyze_app, name="analyze")

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


# ── Analyze commands ──────────────────────────────────────────────────────


@analyze_app.command("paper")
def analyze_paper(
    paper_id: str = typer.Argument(help="Paper arXiv ID (e.g. 2312.00752)"),
    model: str = typer.Option(
        DEFAULT_LLM_MODEL, "--model", "-m",
        help="Ollama model for analysis (e.g. gemma4:31b-cloud)",
    ),
    no_self_correct: bool = typer.Option(
        False, "--no-self-correct", help="Skip self-correction pass",
    ),
    force: bool = typer.Option(False, "--force", help="Re-analyze even if done"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save results to DB"),
) -> None:
    """Analyze a paper: extract 5W1H, strengths/weaknesses, references, evidence.

    Runs the full analysis pipeline via Ollama LLM:
    1. Download + parse paper (if not cached)
    2. 5W1H extraction
    3. Reference chain extraction
    4. Evidence sentence mapping
    5. Self-correction verification

    Args:
        paper_id: Paper arXiv ID (e.g. ``2312.00752``).
        model: Ollama model name for analysis.
        no_self_correct: Skip the self-correction pass.
        force: Re-analyze even if analysis already exists.
        save: Save analysis results to the database.
    """
    from ml_platform.analysis.analyzer import PaperAnalyzer
    from ml_platform.db import PapersDB
    from ml_platform.models import Paper, PaperSource
    from ml_platform.processing.processor import PaperProcessor

    source = "arxiv"
    full_id = f"arxiv_{paper_id}"

    # Check if already analyzed
    db = PapersDB()
    if not force and db.has_analysis(full_id, source):
        existing = db.get_analysis(full_id, source)
        if existing and existing.get("status") == "completed":
            console.print(f"[yellow]Already analyzed: {paper_id}[/] (use --force to re-analyze)")
            console.print(f"  Summary: {existing.get('summary', '')[:200]}")
            return

    # Ensure paper is processed (downloaded + parsed)
    console.print(f"\n[bold blue]📊 Analyzing paper:[/] {paper_id}")
    console.print(f"  Model: {model} | Self-correct: {not no_self_correct}\n")

    paper = Paper(
        paper_id=full_id,
        arxiv_id=paper_id,
        title=paper_id,
        source=PaperSource.ARXIV,
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )

    # Process paper if not already done
    if not paper.parsed_content:
        with console.status("[bold]Downloading & parsing paper..."):
            processor = PaperProcessor()
            proc_result = asyncio.run(processor.process_paper(paper, force=force))

        if not proc_result.success:
            console.print(f"[red]Processing failed: {proc_result.error}[/]")
            raise typer.Exit(1)

        console.print(
            f"  Parsed: {proc_result.extracted.get('chars', 0):,} chars "
            f"({proc_result.extracted.get('extraction_method', '?')})"
        )

    # Run analysis
    def progress(stage: str, pct: float, msg: str) -> None:
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        console.print(f"  [{bar}] {pct:5.1f}% {msg}")

    analyzer = PaperAnalyzer(
        model=model,
        enable_self_correction=not no_self_correct,
        progress_callback=progress,
    )

    try:
        analysis = asyncio.run(analyzer.analyze(paper))
    except Exception as e:
        console.print(f"\n[bold red]❌ Analysis failed:[/] {e}")
        raise typer.Exit(1)

    # Display results
    console.print("\n[bold green]✅ Analysis complete![/]\n")

    # 5W1H
    console.print("[bold cyan]── 5W1H ──[/]")
    for key in ["who", "what", "when", "where", "why", "how"]:
        val = getattr(analysis.five_w1h, key)
        console.print(f"  [bold]{key.upper():>5}[/]  {val[:120]}{'...' if len(val) > 120 else ''}")

    # Strengths / Weaknesses
    console.print("\n[bold cyan]── Strengths ──[/]")
    for s in analysis.sw.strengths:
        console.print(f"  ✅ {s[:100]}{'...' if len(s) > 100 else ''}")

    console.print("\n[bold cyan]── Weaknesses ──[/]")
    for w in analysis.sw.weaknesses:
        console.print(f"  ⚠️  {w[:100]}{'...' if len(w) > 100 else ''}")

    console.print("\n[bold cyan]── Future Work ──[/]")
    for fw in analysis.sw.future_work:
        console.print(f"  🔮 {fw[:100]}{'...' if len(fw) > 100 else ''}")

    # Summary
    console.print(f"\n[bold cyan]── Summary ──[/]")
    console.print(f"  {analysis.summary[:300]}{'...' if len(analysis.summary) > 300 else ''}")

    # Stats
    console.print(f"\n[dim]"
                  f"References: {len(analysis.references)} | "
                  f"Evidence: {len(analysis.evidence)} | "
                  f"Self-corrected: {'yes' if analysis.self_correction_applied else 'no'} | "
                  f"Model: {analysis.model_used}"
                  f"[/dim]")

    # Save to DB
    if save:
        db.save_analysis(full_id, source, analysis)
        console.print(f"[dim]Saved to database[/dim]")

        # Register paper entity
        from ml_platform.analysis.reference_chain import canonical_paper_id
        cid = canonical_paper_id(
            doi=None,
            arxiv_id=paper_id,
            title=getattr(paper, "title", None),
        )
        db.register_entity(cid, None, paper_id, paper.title or paper_id, f"{full_id}/{source}")

        # Register reference entities
        for ref in analysis.references:
            if ref.doi or ref.arxiv_id:
                ref_cid = canonical_paper_id(ref.doi, ref.arxiv_id, ref.title)
                db.register_entity(
                    ref_cid, ref.doi, ref.arxiv_id,
                    ref.title or ref.raw_text[:80],
                    f"{full_id}/{source}",
                )
        console.print(f"[dim]Registered {len(analysis.references)} reference entities[/dim]")


@analyze_app.command("show")
def analyze_show(
    paper_id: str = typer.Argument(help="Paper arXiv ID (e.g. 2312.00752)"),
) -> None:
    """Show existing analysis results for a paper.

    Args:
        paper_id: Paper arXiv ID (e.g. ``2312.00752``).
    """
    from ml_platform.analysis.models import PaperAnalysis
    from ml_platform.db import PapersDB

    db = PapersDB()
    full_id = f"arxiv_{paper_id}"
    analysis = db.get_analysis_object(full_id, "arxiv")

    if analysis is None:
        console.print(f"[yellow]No analysis found for {paper_id}[/]")
        console.print("[dim]Run: ml-research analyze paper {paper_id}[/]")
        return

    console.print(f"\n[bold blue]📊 Analysis:[/] {paper_id}\n")

    console.print("[bold cyan]── 5W1H ──[/]")
    for key in ["who", "what", "when", "where", "why", "how"]:
        val = getattr(analysis.five_w1h, key)
        console.print(f"  [bold]{key.upper():>5}[/]  {val}")

    console.print("\n[bold cyan]── Strengths ──[/]")
    for s in analysis.sw.strengths:
        console.print(f"  ✅ {s}")

    console.print("\n[bold cyan]── Weaknesses ──[/]")
    for w in analysis.sw.weaknesses:
        console.print(f"  ⚠️  {w}")

    console.print("\n[bold cyan]── Future Work ──[/]")
    for fw in analysis.sw.future_work:
        console.print(f"  🔮 {fw}")

    console.print(f"\n[bold cyan]── Summary ──[/]")
    console.print(f"  {analysis.summary}")

    console.print(f"\n[dim]"
                  f"References: {len(analysis.references)} | "
                  f"Evidence: {len(analysis.evidence)} | "
                  f"Model: {analysis.model_used} | "
                  f"Self-corrected: {'yes' if analysis.self_correction_applied else 'no'}"
                  f"[/dim]")


# ── Setup commands ─────────────────────────────────────────────────────


@setup_app.command("deepcode")
def setup_deepcode() -> None:
    """Patch DeepCode package (missing modules + Ollama support)."""
    from ml_platform.codegen.deepcode_setup import run_setup

    run_setup()


if __name__ == "__main__":
    app()


# ── Graph commands ─────────────────────────────────────────────────────


@app.command("graph")
def graph_command(
    ctx: typer.Context,
) -> None:
    """Knowledge graph operations. Use subcommands: build, query, stats, list, export."""
    # If no subcommand, show help
    if ctx.invoked_subcommand is None:
        console.print("[bold]Knowledge Graph Commands:[/]")
        console.print("  ml-research graph build <topic>     Build graph from analyses")
        console.print("  ml-research graph query <topic> <cypher>  Run Cypher query")
        console.print("  ml-research graph stats [topic]     Show graph statistics")
        console.print("  ml-research graph list               List all topic graphs")
        console.print("  ml-research graph export <topic>     Export graph as JSON/MD")


@app.command("graph-build")
def graph_build(
    topic: str = typer.Argument(..., help="Topic name for the graph"),
    papers: str = typer.Option(None, "--papers", "-p", help="Comma-separated paper IDs"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild"),
) -> None:
    """Build a knowledge graph from paper analyses."""
    from ml_platform.graph.builder import GraphBuilder

    paper_ids = [p.strip() for p in papers.split(",")] if papers else None

    builder = GraphBuilder()
    stats = builder.build_topic(topic, paper_ids=paper_ids, force=force)

    console.print(Panel(
        f"[bold]Graph:[/] {topic}\n"
        f"[bold]Nodes:[/] {stats.node_count}  {stats.node_types}\n"
        f"[bold]Edges:[/] {stats.edge_count}  {stats.edge_types}\n"
        f"[bold]Papers:[/] {stats.papers_indexed}\n"
        f"[bold]DB:[/] {stats.db_path}",
        title="Knowledge Graph Built",
    ))


@app.command("graph-query")
def graph_query(
    topic: str = typer.Argument(..., help="Topic name"),
    cypher: str = typer.Argument(..., help="Cypher query string"),
) -> None:
    """Run a Cypher query against a topic graph."""
    from ml_platform.graph.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph.open(topic)
    try:
        results = kg.query(cypher)
        if not results:
            console.print("[dim]No results.[/]")
            return

        table = Table(title=f"Query: {cypher[:60]}")
        if results:
            for key in results[0].keys():
                table.add_column(key)

            for row in results[:50]:
                table.add_row(*[str(v) for v in row.values()])

        console.print(table)
    finally:
        kg.close()


@app.command("graph-stats")
def graph_stats(
    topic: str = typer.Argument(None, help="Topic name (omit for all)"),
) -> None:
    """Show knowledge graph statistics."""
    from ml_platform.graph.builder import GraphBuilder
    from ml_platform.graph.knowledge_graph import KnowledgeGraph

    if topic:
        kg = KnowledgeGraph.open(topic)
        try:
            stats = kg.get_stats()
            _print_stats(stats)
        finally:
            kg.close()
    else:
        graphs = GraphBuilder.list_graphs()
        if not graphs:
            console.print("[dim]No graphs found.[/]")
            return

        for g in graphs:
            if "error" in g:
                console.print(f"  [red]{g['topic']}[/]: {g['error']}")
                continue
            s = g["stats"]
            console.print(
                f"  [bold]{g['topic']}[/]: "
                f"{s['node_count']} nodes, {s['edge_count']} edges, "
                f"{s['papers_indexed']} papers"
            )


@app.command("graph-list")
def graph_list() -> None:
    """List all topic knowledge graphs."""
    from ml_platform.graph.builder import GraphBuilder

    graphs = GraphBuilder.list_graphs()
    if not graphs:
        console.print("[dim]No knowledge graphs found. Use 'graph-build' to create one.[/]")
        return

    table = Table(title="Knowledge Graphs")
    table.add_column("Topic", style="bold")
    table.add_column("Nodes")
    table.add_column("Edges")
    table.add_column("Papers")
    table.add_column("DB Path")

    for g in graphs:
        if "error" in g:
            table.add_row(g["topic"], "[red]ERROR[/]", "", "", g["db_path"])
            continue
        s = g["stats"]
        table.add_row(
            g["topic"],
            str(s["node_count"]),
            str(s["edge_count"]),
            str(s["papers_indexed"]),
            g["db_path"],
        )

    console.print(table)


@app.command("graph-export")
def graph_export(
    topic: str = typer.Argument(..., help="Topic name"),
    fmt: str = typer.Option("json", "--format", "-f", help="Export format: json, md"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
) -> None:
    """Export a knowledge graph."""
    import json as json_mod
    from ml_platform.graph.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph.open(topic)
    try:
        stats = kg.get_stats()
        nodes = kg.get_all_nodes()
        edges = kg.get_all_edges()

        if fmt == "json":
            data = {
                "topic": topic,
                "stats": stats.model_dump(),
                "nodes": nodes,
                "edges": edges,
            }
            content = json_mod.dumps(data, indent=2, ensure_ascii=False)
        elif fmt == "md":
            lines = [f"# Knowledge Graph: {topic}\n"]
            lines.append(f"**Nodes:** {stats.node_count} | **Edges:** {stats.edge_count}\n")
            lines.append("## Nodes\n")
            for n in nodes:
                lines.append(f"- [{n.get('types', ['?'])[0] if isinstance(n.get('types'), list) else n.get('types', '?')}] {n.get('n.label', n.get('value', '?'))}")
            lines.append("\n## Edges\n")
            for e in edges:
                lines.append(f"- {e.get('a.node_id', '?')} --[{e.get('rel', '?')}]--> {e.get('b.node_id', '?')}")
            content = "\n".join(lines)
        else:
            console.print(f"[red]Unknown format: {fmt}. Use json or md.[/]")
            return

        if output:
            Path(output).write_text(content)
            console.print(f"Exported to {output}")
        else:
            console.print(content)
    finally:
        kg.close()


def _print_stats(stats) -> None:
    """Pretty-print graph statistics."""
    console.print(Panel(
        f"[bold]Topic:[/] {stats.topic}\n"
        f"[bold]Nodes:[/] {stats.node_count}\n"
        f"[bold]Edges:[/] {stats.edge_count}\n"
        f"[bold]Papers:[/] {stats.papers_indexed}\n"
        f"[bold]DB:[/] {stats.db_path}\n"
        f"[dim]Node types: {stats.node_types}[/]\n"
        f"[dim]Edge types: {stats.edge_types}[/]",
        title="Graph Stats",
    ))
