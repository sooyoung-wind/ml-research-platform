"""ML Research Platform — CLI interface."""

from __future__ import annotations

import typer
from rich import print as rprint

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


# ── Discover commands ──────────────────────────────────────────────────────


@discover_app.command("search")
def discover_search(
    topic: str = typer.Argument(help="Search topic"),
    top: int = typer.Option(10, "--top", "-n", help="Number of results"),
    source: str = typer.Option("all", "--source", "-s", help="Source: all, arxiv, semantic_scholar, pwc"),
) -> None:
    """Search for papers on a topic."""
    rprint(f"[bold blue]🔍 Searching for:[/] {topic} (top {top}, source: {source})")
    rprint("[dim]Paper discovery pipeline will be implemented in Phase 1[/dim]")


@discover_app.command("daily")
def discover_daily(
    top: int = typer.Option(10, "--top", "-n", help="Number of results"),
) -> None:
    """Run daily paper discovery for configured topics."""
    from ml_platform.config import config

    rprint("[bold blue]📡 Daily Paper Discovery[/]")
    rprint(f"[dim]Topics: {', '.join(config.DEFAULT_TOPICS)}[/dim]")
    rprint("[dim]Daily discovery will be implemented in Phase 1[/dim]")


# ── Process commands ───────────────────────────────────────────────────────


@process_app.command("paper")
def process_paper(
    paper_id: str = typer.Argument(help="Paper ID (arXiv ID, DOI, etc.)"),
) -> None:
    """Process a paper: download PDF, parse, enrich metadata."""
    rprint(f"[bold blue]📄 Processing paper:[/] {paper_id}")
    rprint("[dim]Paper processing pipeline will be implemented in Phase 2[/dim]")


# ── Codegen commands ───────────────────────────────────────────────────────


@codegen_app.command("generate")
def codegen_generate(
    paper_id: str = typer.Argument(help="Paper ID"),
    engine: str = typer.Option("papercoder", "--engine", "-e", help="Engine: papercoder, deepcode"),
) -> None:
    """Generate code from a paper."""
    rprint(f"[bold blue]⚙️ Generating code for:[/] {paper_id} (engine: {engine})")
    rprint("[dim]Code generation will be implemented in Phase 3[/dim]")


# ── Status command ─────────────────────────────────────────────────────────


@app.command("status")
def status() -> None:
    """Show platform status and database stats."""
    from ml_platform.db import PapersDB

    db = PapersDB()
    stats = db.get_stats()

    rprint("[bold]ML Research Platform Status[/]")
    rprint(f"  Database: [green]{db.db_path}[/]")
    rprint(f"  Total papers: [bold]{stats['total_papers']}[/]")
    rprint(f"  With code:    {stats['with_code']}")
    rprint(f"  Without code: {stats['without_code']}")
    rprint(f"  By status:    {stats['by_status']}")


@app.command("version")
def version() -> None:
    """Show version."""
    rprint("[bold]ml-research-platform[/] v0.1.0")


if __name__ == "__main__":
    app()
