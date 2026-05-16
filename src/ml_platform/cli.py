"""ML Research Platform — CLI interface."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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

console = Console()


# ── Discover commands ──────────────────────────────────────────────────────


@discover_app.command("search")
def discover_search(
    topic: str = typer.Argument(help="Search topic"),
    top: int = typer.Option(10, "--top", "-n", help="Number of results"),
    source: str = typer.Option("all", "--source", "-s", help="Source: all, arxiv, semantic_scholar, huggingface"),
) -> None:
    """Search for papers on a topic across multiple sources."""
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
    """Run daily paper discovery for all configured topics."""
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
    """Show today's trending papers from HuggingFace."""
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
        stars = "-"  # trending doesn't include stars directly
        table.add_row(str(i), str(paper.upvotes), title, code, stars)

    console.print(table)
    console.print(f"\n[dim]Showing {len(papers)} trending papers from HuggingFace[/dim]")


# ── Process commands ───────────────────────────────────────────────────────


@process_app.command("paper")
def process_paper(
    paper_id: str = typer.Argument(help="Paper arXiv ID (e.g. 2312.00752)"),
    no_download: bool = typer.Option(False, "--no-download", help="Skip PDF download"),
    no_parse: bool = typer.Option(False, "--no-parse", help="Skip GROBID parsing"),
    no_enrich: bool = typer.Option(False, "--no-enrich", help="Skip metadata enrichment"),
    force: bool = typer.Option(False, "--force", help="Re-process even if already done"),
) -> None:
    """Process a paper: download PDF, parse with GROBID, enrich metadata."""
    import time as _time
    from ml_platform.models import Paper, PaperSource

    console.print(f"\n[bold blue]📄 Processing paper:[/] {paper_id}\n")

    # Create a Paper object from the ID
    paper = Paper(
        paper_id=f"arxiv_{paper_id}",
        arxiv_id=paper_id,
        title=paper_id,  # will be filled by enrichment
        source=PaperSource.ARXIV,
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )

    from ml_platform.processing.pipeline import ProcessingPipeline
    pipeline = ProcessingPipeline()

    start = _time.time()
    result = asyncio.run(pipeline.process_paper(
        paper,
        download=not no_download,
        parse=not no_parse,
        enrich=not no_enrich,
        force=force,
    ))

    # Display result
    if result.success:
        console.print("[bold green]✅ Processing complete![/]")
    else:
        console.print(f"[bold red]❌ Processing failed:[/] {result.error}")

    if result.download:
        dl = result.download
        if dl.get("skipped"):
            console.print(f"  [dim]PDF: skipped (already downloaded)[/]")
        elif dl.get("success"):
            size_kb = (dl.get("size_bytes", 0) or 0) / 1024
            console.print(f"  PDF: {dl.get('path')} ({size_kb:.0f} KB)")
        else:
            console.print(f"  [red]PDF download failed: {dl.get('error')}[/]")

    if result.parsed:
        pr = result.parsed
        if pr.get("skipped"):
            console.print(f"  [dim]Parse: skipped (already parsed)[/]")
        elif pr.get("success"):
            console.print(
                f"  Parse: {pr.get('sections', 0)} sections, "
                f"{pr.get('references', 0)} refs, "
                f"{pr.get('figures', 0)} figures"
            )
            if pr.get("partial"):
                console.print(f"  [yellow]Parse had errors: {pr.get('errors')}[/]")

    if result.enriched:
        console.print(f"  Enriched: citations={paper.citation_count}, code={'yes' if paper.has_code else 'no'}")

    console.print(f"  [dim]Completed in {result.duration:.1f}s[/dim]")


@process_app.command("batch")
def process_batch(
    topic: str = typer.Argument(help="Search topic to discover and process"),
    top: int = typer.Option(5, "--top", "-n", help="Number of papers to process"),
    no_download: bool = typer.Option(False, "--no-download"),
    no_parse: bool = typer.Option(False, "--no-parse"),
    no_enrich: bool = typer.Option(False, "--no-enrich"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Discover papers on a topic and process them all."""
    from ml_platform.discovery.pipeline import DiscoveryPipeline
    from ml_platform.processing.pipeline import ProcessingPipeline

    console.print(f"\n[bold blue]🔄 Batch processing:[/] {topic} (top {top})\n")

    # Step 1: Discover
    with console.status("[bold]Discovering papers..."):
        discovery = DiscoveryPipeline()
        result = asyncio.run(discovery.search(query=topic, top_n=top))

    console.print(f"  Found {result.total_found} papers, processing top {len(result.papers)}\n")

    # Step 2: Process
    pipeline = ProcessingPipeline()
    results = asyncio.run(pipeline.process_batch(
        result.papers,
        download=not no_download,
        parse=not no_parse,
        enrich=not no_enrich,
        force=force,
    ))

    # Summary
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
            healthy = await client.check_health()
            return healthy

    healthy = asyncio.run(check())

    if healthy:
        console.print("[bold green]✅ GROBID is running[/] (http://localhost:8070)")
    else:
        console.print("[bold red]❌ GROBID is not responding[/]")
        console.print("[dim]Start with: docker run -d --name grobid --rm -p 8070:8070 lfoppiano/grobid:0.8.1[/dim]")


# ── Codegen commands ───────────────────────────────────────────────────────


@codegen_app.command("generate")
def codegen_generate(
    paper_id: str = typer.Argument(help="Paper ID"),
    engine: str = typer.Option("papercoder", "--engine", "-e", help="Engine: papercoder, deepcode"),
) -> None:
    """Generate code from a paper."""
    console.print(f"[bold blue]⚙️ Generating code for:[/] {paper_id} (engine: {engine})")
    console.print("[dim]Code generation will be implemented in Phase 3[/dim]")


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


if __name__ == "__main__":
    app()
