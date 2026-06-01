"""
命令行入口 — 框架的统一启动点。

使用方式：
  vma "task description"                             # Ollama + Qwen3.5:4b（默认）
  vma "task" --backend mock                          # mock 模式，零依赖
  vma "task" --backend ollama --model qwen3.5:4b     # Ollama 本地推理
  vma "task" --backend vllm --model Qwen2.5-7B       # vLLM 本地推理
  vma "task" --backend deepseek --model ...          # 云端 API（过渡期）
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from verifiable_multi_agent.backends import DeepSeekBackend, OllamaBackend, OpenAICompatibleBackend
from verifiable_multi_agent.orchestrator import Orchestrator
from verifiable_multi_agent.profiler import LlmProfiler
from verifiable_multi_agent.reporting import format_contract_report
from verifiable_multi_agent.trace import build_run_trace, run_trace_json, save_run_trace

app = typer.Typer(help="Run the verifiable multi-agent research scaffold.")
console = Console(no_color=True)


def load_env_file(path: Path = Path(".env")) -> None:
    """加载 .env 文件到环境变量，支持 ${VAR} 引用已有环境变量。"""
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
    task: str = typer.Argument(..., help="Task for the multi-agent scaffold."),
    memory: Path = typer.Option(Path("data/protocol_memory.jsonl"), help="Protocol memory JSONL path."),
    backend: str = typer.Option("ollama", help="Backend: mock, ollama, vllm, or deepseek."),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL."),
    model: str = typer.Option("local-model", help="Model name for vLLM or DeepSeek."),
    profile_model: str | None = typer.Option(None, help="Small model for LLM-based task profiling (default: qwen3.5:0.8b for ollama)."),
    judge_model: str | None = typer.Option(None, help="Optional stronger model for verifier/synthesizer."),
    api_key: str | None = typer.Option(None, help="API key. Defaults to DEEPSEEK_API_KEY for DeepSeek."),
    hide_trace: bool = typer.Option(False, help="Hide internal topology, execution summary, and contract trace."),
    json_trace: bool = typer.Option(False, "--json-trace", help="Output the complete run trace as JSON."),
    contract_report: bool = typer.Option(False, "--contract-report", help="Output a human-readable contract report."),
    save_run: bool = typer.Option(False, "--save-run", help="Save runs/{run_id}/trace.json."),
    router: str = typer.Option("legacy", "--router", help="Router mode: legacy, rule, or quant."),
) -> None:
    load_env_file()
    llm = None
    profiler = None
    if backend == "ollama":
        ollama_model = model if model != "local-model" else "qwen3.5:4b"
        llm = OllamaBackend(model=ollama_model)
        # 用独立的小模型做任务画像，0.8b 足够准确且极快
        profiler_model_name = profile_model or "qwen3.5:0.8b"
        profiler = LlmProfiler(OllamaBackend(model=profiler_model_name))
    elif backend == "vllm":
        llm = OpenAICompatibleBackend(base_url=base_url, model=model)
    elif backend == "deepseek":
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise typer.BadParameter("DeepSeek backend requires --api-key or DEEPSEEK_API_KEY.")
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        deepseek_model = model if model != "local-model" else os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        deepseek_judge_model = judge_model or os.getenv("DEEPSEEK_JUDGE_MODEL")
        llm = DeepSeekBackend(
            api_key=key,
            model=deepseek_model,
            judge_model=deepseek_judge_model,
            base_url=deepseek_base_url,
        )
    elif backend != "mock":
        raise typer.BadParameter("backend must be one of: mock, ollama, vllm, deepseek.")
    if router not in {"legacy", "rule", "quant"}:
        raise typer.BadParameter("router must be one of: legacy, rule, quant.")
    trace = Orchestrator(memory_path=memory, backend=llm, profiler=profiler, router_mode=router).solve(task)
    run_trace = build_run_trace(trace)
    saved_path = save_run_trace(run_trace) if save_run else None

    if json_trace:
        typer.echo(run_trace_json(run_trace))
        return

    if contract_report:
        console.print(format_contract_report(trace))
        if saved_path:
            console.print(f"\nSaved run: {saved_path}")
        return

    zh = _contains_cjk(task)
    labels = _labels(zh)
    console.print(f"[bold]{labels['answer']}:[/bold] {trace.final_answer}")
    if saved_path:
        console.print(f"[bold]Saved run:[/bold] {saved_path}")
    if hide_trace:
        return

    console.rule(labels["debug_trace"])
    console.print(f"[bold]{labels['topology']}:[/bold] {trace.topology.value}")
    console.print(f"[bold]{labels['topology_reason']}:[/bold] {trace.topology_reason}")
    console.print(f"[bold]{labels['complexity']}:[/bold] {trace.profile.complexity}")
    if trace.metadata.get("topology_spec"):
        _print_quant_router(trace.metadata["topology_spec"])
    console.print(f"[bold]{labels['accepted']}:[/bold] {trace.verification.accepted if trace.verification else False}")
    console.print(f"[bold]{labels['execution_summary']}:[/bold]")
    for item in trace.execution_summary:
        console.print(f"- {item}")

    table = Table(title=labels["contract_trace"], box=box.ASCII)
    table.add_column(labels["role"])
    table.add_column(labels["subtask"])
    table.add_column(labels["support"])
    table.add_column(labels["action"])
    for message in trace.messages:
        support = "是" if zh and message.has_support else "否" if zh else "yes" if message.has_support else "no"
        table.add_row(message.role.value, message.subtask, support, message.action)
    console.print(table)


def _print_quant_router(topology_spec: dict) -> None:
    needs = [name for name, enabled in topology_spec["capability_needs"].items() if enabled]
    console.print("[bold]TCI:[/bold] " + f"{topology_spec['tci']:.3f}")
    console.print("[bold]capability_needs:[/bold] " + ", ".join(needs))
    console.print("[bold]generation_reasons:[/bold]")
    for reason in topology_spec["generation_reasons"]:
        console.print(f"- {reason}")

def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _labels(zh: bool) -> dict[str, str]:
    if not zh:
        return {
            "topology": "Topology",
            "topology_reason": "Topology reason",
            "complexity": "Complexity",
            "accepted": "Accepted",
            "answer": "Answer",
            "execution_summary": "Execution summary",
            "contract_trace": "Contract Trace",
            "role": "Role",
            "subtask": "Subtask",
            "support": "Support",
            "action": "Action",
            "debug_trace": "Debug Trace",
        }
    return {
        "topology": "拓扑",
        "topology_reason": "拓扑原因",
        "complexity": "复杂度",
        "accepted": "验证通过",
        "answer": "答案",
        "execution_summary": "执行摘要",
        "contract_trace": "合约轨迹",
        "role": "角色",
        "subtask": "子任务",
        "support": "支撑",
        "action": "动作",
        "debug_trace": "调试轨迹",
    }


if __name__ == "__main__":
    app()
