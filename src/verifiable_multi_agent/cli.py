from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from verifiable_multi_agent.backends import OpenAICompatibleBackend
from verifiable_multi_agent.orchestrator import Orchestrator

app = typer.Typer(help="Run the verifiable multi-agent research scaffold.")
console = Console(no_color=True)


@app.command()
def solve(
    task: str = typer.Argument(..., help="Task for the multi-agent scaffold."),
    memory: Path = typer.Option(Path("data/protocol_memory.jsonl"), help="Protocol memory JSONL path."),
    backend: str = typer.Option("mock", help="Backend: mock or vllm."),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL."),
    model: str = typer.Option("local-model", help="Model name served by vLLM."),
) -> None:
    llm = None
    if backend == "vllm":
        llm = OpenAICompatibleBackend(base_url=base_url, model=model)
    trace = Orchestrator(memory_path=memory, backend=llm).solve(task)
    console.print(f"[bold]Topology:[/bold] {trace.topology.value}")
    console.print(f"[bold]Complexity:[/bold] {trace.profile.complexity}")
    console.print(f"[bold]Accepted:[/bold] {trace.verification.accepted if trace.verification else False}")
    console.print(f"[bold]Answer:[/bold] {trace.final_answer}")

    table = Table(title="Contract Trace", box=box.ASCII)
    table.add_column("Role")
    table.add_column("Subtask")
    table.add_column("Support")
    table.add_column("Action")
    for message in trace.messages:
        table.add_row(message.role.value, message.subtask, "yes" if message.has_support else "no", message.action)
    console.print(table)


if __name__ == "__main__":
    app()
