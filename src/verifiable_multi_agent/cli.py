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
    judge_model: str | None = typer.Option(None, help="Optional stronger model for verifier/synthesizer."),
    api_key: str | None = typer.Option(None, help="API key. Defaults to DEEPSEEK_API_KEY for DeepSeek."),
) -> None:
    load_env_file()
    llm = None
    if backend == "ollama":
        ollama_model = model if model != "local-model" else "qwen3.5:4b"
        llm = OllamaBackend(model=ollama_model)
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
