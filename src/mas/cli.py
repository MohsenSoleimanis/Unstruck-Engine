"""CLI entry point for the Multi-Agent System."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(name="mas", help="Multi-Agent System CLI")
console = Console()


@app.command()
def run(
    query: str = typer.Argument(help="The query to process"),
    context_file: str | None = typer.Option(None, "--context", "-c", help="Path to context JSON file"),
    file: str | None = typer.Option(None, "--file", "-f", help="Path to document to process"),
    max_iterations: int = typer.Option(3, "--max-iter", "-m", help="Max orchestrator iterations"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Run a query through the multi-agent pipeline."""
    from mas.pipeline import MASPipeline

    context = {}
    if context_file:
        context = json.loads(Path(context_file).read_text())
    if file:
        context["file_path"] = file

    pipeline = MASPipeline()

    with console.status("[bold green]Running multi-agent pipeline..."):
        result = asyncio.run(pipeline.run(query=query, context=context, max_iterations=max_iterations))

    # Display results
    console.print(Panel(json.dumps(result, indent=2, default=str)[:3000], title="Results"))

    # Cost summary
    cost = pipeline.cost_tracker.get_summary()
    table = Table(title="Cost Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total Tokens", str(cost["session"]["total_tokens"]))
    table.add_row("Total Cost", f"${cost['session']['total_cost_usd']:.4f}")
    table.add_row("LLM Calls", str(cost["session"]["num_calls"]))
    console.print(table)

    if output:
        Path(output).write_text(json.dumps(result, indent=2, default=str))
        console.print(f"[green]Output saved to {output}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    workers: int = typer.Option(1, help="Number of workers"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "mas.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


@app.command()
def agents():
    """List all registered agents."""
    from mas.agents.registry import registry

    # Trigger agent registration by importing
    import mas.agents.protocol  # noqa: F401
    import mas.agents.rag  # noqa: F401
    import mas.agents.research  # noqa: F401

    table = Table(title="Registered Agents")
    table.add_column("Type", style="cyan")
    table.add_column("Description")
    table.add_column("Version", style="green")

    for agent in registry.list_agents():
        table.add_row(agent["agent_type"], agent["description"], agent["version"])

    console.print(table)


@app.command()
def cost_report(output: str = typer.Option("./output/cost_report.json", help="Output path")):
    """Export cost tracking report."""
    from mas.pipeline import MASPipeline

    pipeline = MASPipeline()
    path = pipeline.cost_tracker.export(Path(output))
    console.print(f"[green]Cost report exported to {path}")


if __name__ == "__main__":
    app()
