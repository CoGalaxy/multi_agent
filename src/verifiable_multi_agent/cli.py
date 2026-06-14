"""Command line entry for the quant-only thesis prototype."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from verifiable_multi_agent.backends import DeepSeekBackend, OllamaBackend, OpenAICompatibleBackend
from verifiable_multi_agent.orchestrator import Orchestrator
from verifiable_multi_agent.profiler import LlmProfiler
from verifiable_multi_agent.reporting import format_contract_report
from verifiable_multi_agent.routing_memory import ProtocolMemory
from verifiable_multi_agent.trace import build_run_trace, run_trace_json, save_run_trace

app = typer.Typer(help="Run the quant-only verifiable multi-agent prototype.")
console = Console(no_color=True)


def load_env_file(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from .env without overwriting existing env vars."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value.startswith("${") and value.endswith("}"):
            value = os.getenv(value[2:-1], "")
        os.environ.setdefault(key, value)


@app.command()
def solve(
    task: str = typer.Argument(..., help="Task for the multi-agent prototype."),
    memory: Path = typer.Option(Path("data/protocol_memory.jsonl"), help="Protocol memory JSONL path."),
    backend: str = typer.Option("ollama", help="Backend: ollama, vllm, or deepseek."),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL for vLLM."),
    model: str = typer.Option("local-model", help="Model name for vLLM, Ollama, or DeepSeek."),
    profile_model: str | None = typer.Option(None, help="Small Ollama model for task profiling."),
    judge_model: str | None = typer.Option(None, help="Optional DeepSeek judge model for verifier/synthesizer."),
    api_key: str | None = typer.Option(None, help="API key. Defaults to DEEPSEEK_API_KEY for DeepSeek."),
    hide_trace: bool = typer.Option(False, help="Hide debug trace in default text output."),
    json_trace: bool = typer.Option(False, "--json-trace", help="Output the complete run trace as JSON."),
    contract_report: bool = typer.Option(False, "--contract-report", help="Output a human-readable contract report."),
    save_run: bool = typer.Option(False, "--save-run", help="Save runs/{run_id}/trace.json."),
    router: str = typer.Option("quant", "--router", help="Router mode: quant."),
    show_topology: bool = typer.Option(False, "--show-topology", help="Show generated graph topology."),
    rag_corpus: Path = typer.Option(Path("data/rag_corpus.jsonl"), "--rag-corpus", help="Local JSONL corpus for RAG evidence retrieval."),
    routing_memory: bool = typer.Option(True, "--routing-memory/--no-routing-memory", help="Enable route memory correction."),
) -> None:
    load_env_file()
    if router != "quant":
        raise typer.BadParameter("only quant router is kept in the thesis main line.")

    llm, profiler = _build_backend(backend, base_url, model, profile_model, judge_model, api_key)
    route_memory = ProtocolMemory("runs/memory.json") if routing_memory else None
    trace = Orchestrator(
        memory_path=memory,
        backend=llm,
        profiler=profiler,
        router_mode="quant",
        rag_corpus_path=rag_corpus,
        routing_memory=route_memory,
    ).solve(task)
    run_trace = build_run_trace(trace)
    saved_path = save_run_trace(run_trace) if save_run else None

    if json_trace:
        _write_utf8(run_trace_json(run_trace) + "\n")
        return

    if contract_report:
        if show_topology:
            _print_generated_topology(trace)
        console.print(format_contract_report(trace), markup=False)
        if saved_path:
            console.print(f"\nSaved run: {saved_path}")
        return

    console.print("[bold]Answer:[/bold]")
    console.print(trace.final_answer or "", markup=False)
    if saved_path:
        console.print(f"[bold]Saved run:[/bold] {saved_path}")
    if hide_trace:
        return

    console.rule("Debug Trace")
    console.print(f"[bold]Topology:[/bold] {trace.topology.value}")
    console.print(f"[bold]Topology reason:[/bold] {trace.topology_reason}")
    console.print(
        "[bold]TaskProfile:[/bold] "
        f"complexity={trace.profile.complexity:.2f}, "
        f"verifiability={trace.profile.verifiability:.2f}"
    )
    if show_topology:
        _print_generated_topology(trace)
    if trace.metadata.get("topology_spec"):
        _print_quant_router(trace.metadata["topology_spec"])
    console.print(f"[bold]Accepted:[/bold] {trace.verification.accepted if trace.verification else False}")
    console.print("[bold]Execution summary:[/bold]")
    for item in trace.execution_summary:
        console.print(f"- {item}")
    _print_contract_table(trace.messages)


def _build_backend(
    backend: str,
    base_url: str,
    model: str,
    profile_model: str | None,
    judge_model: str | None,
    api_key: str | None,
):
    if backend == "ollama":
        ollama_model = model if model != "local-model" else "qwen3.5:4b"
        llm = OllamaBackend(model=ollama_model)
        profiler = LlmProfiler(OllamaBackend(model=profile_model or "qwen3.5:0.8b"))
        return llm, profiler
    if backend == "vllm":
        return OpenAICompatibleBackend(base_url=base_url, model=model), None
    if backend == "deepseek":
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise typer.BadParameter("DeepSeek backend requires --api-key or DEEPSEEK_API_KEY.")
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        deepseek_model = model if model != "local-model" else os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        return (
            DeepSeekBackend(
                api_key=key,
                model=deepseek_model,
                judge_model=judge_model or os.getenv("DEEPSEEK_JUDGE_MODEL"),
                base_url=deepseek_base_url,
            ),
            None,
        )
    raise typer.BadParameter("backend must be one of: ollama, vllm, deepseek.")


def _write_utf8(text: str) -> None:
    """Write machine-readable output without Windows console encoding loss."""
    sys.stdout.buffer.write(text.encode("utf-8"))


def _print_quant_router(topology_spec: dict) -> None:
    needs = [name for name, enabled in topology_spec["capability_needs"].items() if enabled]
    console.print(
        "[bold]Profile:[/bold] "
        f"complexity={topology_spec['complexity']:.2f}, "
        f"verifiability={topology_spec['verifiability']:.2f}"
    )
    console.print("[bold]capability_needs:[/bold] " + ", ".join(needs))
    console.print("[bold]generation_reasons:[/bold]")
    for reason in topology_spec["generation_reasons"]:
        console.print(f"- {reason}")


def _print_generated_topology(trace) -> None:
    generated = trace.metadata.get("generated_topology")
    execution = trace.metadata.get("graph_execution")
    if not generated:
        return
    console.print("[bold][Generated Topology][/bold]")
    node_labels = [node["label"] or node["type"] for node in generated.get("nodes", [])]
    console.print(" -> ".join(node_labels) if node_labels else "(blocked)")
    console.print(f"blocked={generated.get('blocked')}")
    if generated.get("block_reason"):
        console.print(f"block_reason={generated.get('block_reason')}")
    if generated.get("validation_errors"):
        console.print(f"validation_errors={generated.get('validation_errors')}")
    if execution:
        console.print("[bold][Graph Execution][/bold]")
        console.print(f"executed_nodes={execution.get('executed_nodes', [])}")
        console.print(f"skipped_nodes={execution.get('skipped_nodes', [])}")
        console.print(f"review_loops_used={execution.get('review_loops_used', 0)}")
        console.print(f"execution_mode={execution.get('execution_mode')}")


def _print_contract_table(messages) -> None:
    table = Table(title="Contract Trace", box=box.ASCII)
    table.add_column("Role")
    table.add_column("Subtask")
    table.add_column("Support")
    table.add_column("Action")
    for message in messages:
        table.add_row(
            message.role.value,
            message.subtask,
            "yes" if message.has_support else "no",
            message.action,
        )
    console.print(table)


if __name__ == "__main__":
    app()
