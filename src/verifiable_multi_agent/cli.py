"""
命令行入口 — 框架的统一启动点。

使用方式：
  vma "task description"                        # mock 模式，零依赖
  vma "task" --backend deepseek --model ...     # 云端 API（过渡期）
  vma "task" --backend vllm --model Qwen2.5-7B  # 本地模型（最终配置）
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from verifiable_multi_agent.backends import DeepSeekBackend, OpenAICompatibleBackend
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
    backend: str = typer.Option("mock", help="Backend: mock, vllm, or deepseek."),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL."),
    model: str = typer.Option("local-model", help="Model name for vLLM or DeepSeek."),
    judge_model: str | None = typer.Option(None, help="Optional stronger model for verifier/synthesizer."),
    api_key: str | None = typer.Option(None, help="API key. Defaults to DEEPSEEK_API_KEY for DeepSeek."),
) -> None:
    load_env_file()
    llm = None
    if backend == "vllm":
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
        raise typer.BadParameter("backend must be one of: mock, vllm, deepseek.")
    trace = Orchestrator(memory_path=memory, backend=llm).solve(task)
    zh = _contains_cjk(task)
    labels = _labels(zh)
    console.print(f"[bold]{labels['topology']}:[/bold] {trace.topology.value}")
    console.print(f"[bold]{labels['topology_reason']}:[/bold] {trace.topology_reason}")
    console.print(f"[bold]{labels['complexity']}:[/bold] {trace.profile.complexity}")
    console.print(f"[bold]{labels['accepted']}:[/bold] {trace.verification.accepted if trace.verification else False}")
    console.print(f"[bold]{labels['answer']}:[/bold] {trace.final_answer}")
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
    }


if __name__ == "__main__":
    app()
