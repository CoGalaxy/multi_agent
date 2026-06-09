"""scripts/ 目录的 pytest 配置 — 提供 --backend 选项和结果收集。"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest

# 确保项目根目录在 path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_dotenv() -> None:
    """加载项目根目录的 .env 文件到环境变量。"""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value.startswith("${") and value.endswith("}"):
            value = os.environ.get(value[2:-1], "")
        os.environ.setdefault(key, value)


_load_dotenv()


@dataclass
class TaskResult:
    task_id: str
    task_type: str
    backend: str
    task_text: str
    complexity: float
    verifiability: float
    topology_used: str
    accepted: bool
    support_rate: float
    claim: str = ""
    evidence_count: int = 0


@dataclass
class ResultCollector:
    results: list[TaskResult] = field(default_factory=list)

    def add(self, r: TaskResult) -> None:
        self.results.append(r)

    def csv_header(self) -> str:
        return (
            "task_id,task_type,backend,complexity,verifiability,"
            "topology_used,accepted,support_rate,claim_preview,evidence_count"
        )

    def csv_line(self, r: TaskResult) -> str:
        claim_short = r.claim[:80].replace('"', "'") if r.claim else ""
        return (
            f"{r.task_id},{r.task_type},{r.backend},"
            f"{r.complexity:.3f},{r.verifiability:.3f},"
            f"{r.topology_used},{r.accepted},{r.support_rate:.3f},"
            f'"{claim_short}",{r.evidence_count}'
        )

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write(self.csv_header() + "\n")
            for r in self.results:
                f.write(self.csv_line(r) + "\n")

    def summary(self) -> str:
        if not self.results:
            return "No results collected."
        total = len(self.results)
        single = sum(1 for r in self.results if "SINGLE" in r.topology_used.upper())
        supervisor = sum(1 for r in self.results if "SUPERVISOR" in r.topology_used.upper())
        review = sum(1 for r in self.results if "REVIEW" in r.topology_used.upper())
        accepted_count = sum(1 for r in self.results if r.accepted)
        avg_support = sum(r.support_rate for r in self.results) / total if total else 0.0
        avg_complexity = sum(r.complexity for r in self.results) / total if total else 0.0
        avg_verifiability = sum(r.verifiability for r in self.results) / total if total else 0.0

        lines = [
            "=" * 60,
            "SMOKE TEST SUMMARY",
            "=" * 60,
            f"Total runs: {total}",
            f"SINGLE_AGENT:      {single} ({single/total*100:.0f}%)" if total else "",
            f"SUPERVISOR_WORKER: {supervisor} ({supervisor/total*100:.0f}%)" if total else "",
            f"REVIEW_LOOP:       {review} ({review/total*100:.0f}%)" if total else "",
            f"Accepted:          {accepted_count}/{total} ({accepted_count/total*100:.0f}%)" if total else "",
            f"Avg support_rate:  {avg_support:.3f}",
            f"Avg complexity:    {avg_complexity:.3f}",
            f"Avg verifiability: {avg_verifiability:.3f}",
        ]
        for backend in sorted({r.backend for r in self.results}):
            br = [r for r in self.results if r.backend == backend]
            bt = len(br)
            if bt == 0:
                continue
            ba = sum(1 for r in br if r.accepted)
            lines.append(
                f"[{backend}] {ba}/{bt} accepted, "
                f"avg support={sum(r.support_rate for r in br)/bt:.3f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ── 全局 collector ────────────────────────────────────────────────

_collector = ResultCollector()


@pytest.fixture(scope="session")
def collector() -> ResultCollector:
    return _collector


# ── CLI 选项 ──────────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--backend",
        action="store",
        default="deepseek",
        help="Backend: deepseek, qwen, or all",
    )


# ── 动态 backend parametrization ──────────────────────────────────


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """根据 --backend 参数动态生成 backend_name fixture 的取值。"""
    if "backend_name" in metafunc.fixturenames:
        val = metafunc.config.getoption("--backend", default="deepseek")
        if val == "all":
            names: list[str] = []
            for name in ("deepseek", "qwen"):
                try:
                    _ = _try_make_backend(name)
                    names.append(name)
                except pytest.skip.Exception:
                    pass
            metafunc.parametrize("backend_name", names or ["deepseek"])
        else:
            metafunc.parametrize("backend_name", [val])


def _try_make_backend(name: str) -> None:
    """尝试创建 backend 以检查可用性（不实际使用）。"""
    import os as _os
    import httpx as _httpx

    if name == "deepseek":
        if not _os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")
    elif name == "qwen":
        key = _os.getenv("QWEN_API_KEY")
        if key:
            return  # 有 key 就认为可用
        ollama_url = _os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            r = _httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
            r.raise_for_status()
        except Exception:
            pytest.skip("Neither QWEN_API_KEY nor local Ollama available")


# ── Session 收尾 ──────────────────────────────────────────────────


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _collector.results:
        return
    summary = _collector.summary()
    # 用 session 的终端报告器输出
    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal:
        terminal.write_line(summary)
    else:
        print(summary)
    output_path = _PROJECT_ROOT / "runs" / "smoke_results.csv"
    _collector.write_csv(output_path)
    if terminal:
        terminal.write_line(f"Results written to {output_path}")
    else:
        print(f"Results written to {output_path}")
