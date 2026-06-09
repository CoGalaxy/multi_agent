#!/usr/bin/env python3
"""端到端冒烟测试 — 使用真实 LLM 后端验证 10 条中文任务的画像→路由→执行全流程。

用法:
  pytest scripts/smoke_test.py -v --backend deepseek
  pytest scripts/smoke_test.py -v --backend qwen
  pytest scripts/smoke_test.py -v --backend all
  pytest scripts/smoke_test.py -v -k "T01"

环境变量:
  DEEPSEEK_API_KEY  — deepseek 后端必需
  QWEN_API_KEY      — qwen 云端 API（可选，无则尝试本地 Ollama）
  OLLAMA_BASE_URL   — Ollama 端点（默认 http://localhost:11434）
  OLLAMA_MODEL      — Qwen 模型名（默认 qwen3.5:4b）
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import httpx
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.conftest import ResultCollector, TaskResult  # noqa: E402

from verifiable_multi_agent.backends import (  # noqa: E402
    DeepSeekBackend,
    LlmBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
)
from verifiable_multi_agent.memory import ProtocolMemory  # noqa: E402
from verifiable_multi_agent.orchestrator import Orchestrator  # noqa: E402
from verifiable_multi_agent.profiler import LlmProfiler  # noqa: E402

# ── 任务集（10 条，覆盖 4+1 种类型）──────────────────────────────

TASKS: list[tuple[str, str, str]] = [
    # 事实比较类（期望：SUPERVISOR 或 REVIEW）
    ("T01", "比较 AutoGen 和 LangGraph 在多 Agent 编排上的核心架构差异，并给出各自适用场景。", "factual_compare"),
    ("T02", "比较 Transformer 和 Mamba 架构在长序列建模上的效率差异，列出支撑证据。", "factual_compare"),
    # 多跳推理类（期望：SUPERVISOR 或 REVIEW）
    ("T03", "DeepSeek-v4 的训练数据截止时间是什么？这对它回答 2025 年事件的能力有何影响？", "multi_hop"),
    ("T04", "如果一个任务需要调用三个工具、结果之间存在依赖关系，应该选择什么拓扑结构，为什么？", "multi_hop"),
    # 开放分析类（期望：SINGLE 或 SUPERVISOR）
    ("T05", "分析大型语言模型在法律文书生成场景中的主要风险，并提出缓解措施。", "open_analysis"),
    ("T06", "说明为什么多 Agent 系统中的幻觉传播比单 Agent 更难被发现。", "open_analysis"),
    # 验证挑战类（期望：REVIEW，高 verifiability）
    ("T07", "请验证以下说法是否正确：RAG 系统可以完全消除 LLM 的幻觉问题。要求给出反驳证据。", "verification"),
    ("T08", "请核查：MetaGPT 是否使用了 ContractMessage 机制？如果没有，它用什么替代？", "verification"),
    # 路由边界类
    ("T09", "用一句话总结什么是 Chain-of-Thought 提示。", "boundary_simple"),
    ("T10", "设计一个包含 Planner、Executor、Verifier 三个角色的任务流，"
             "验证 RAG 在医疗问答场景中的证据充分性，并给出改进建议。", "boundary_complex"),
]


# ── Backend 工厂 ──────────────────────────────────────────────────

def _make_deepseek_backend() -> LlmBackend:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    return DeepSeekBackend(api_key=key, model=model, base_url=base_url, timeout=180.0)


def _make_qwen_backend() -> LlmBackend:
    key = os.getenv("QWEN_API_KEY")
    if key:
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.getenv("QWEN_MODEL", "qwen-plus")
        return OpenAICompatibleBackend(base_url=base_url, model=model, api_key=key, timeout=180.0)
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        r.raise_for_status()
        model = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        return OllamaBackend(model=model, base_url=ollama_url, timeout=180.0)
    except Exception:
        pytest.skip("Neither QWEN_API_KEY nor local Ollama available")


# ── Session fixtures ──────────────────────────────────────────────


@pytest.fixture
def backend_name(request: pytest.FixtureRequest) -> str:
    """由 pytest_generate_tests 动态 parametrize。"""
    return request.param


@pytest.fixture
def llm_backend(backend_name: str) -> LlmBackend:
    if backend_name == "deepseek":
        return _make_deepseek_backend()
    return _make_qwen_backend()


@pytest.fixture
def profiler(llm_backend: LlmBackend) -> LlmProfiler | None:
    try:
        if isinstance(llm_backend, DeepSeekBackend):
            key = os.getenv("DEEPSEEK_API_KEY", "")
            base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            return LlmProfiler(
                DeepSeekBackend(api_key=key, model="deepseek-v4-flash", base_url=base, timeout=120.0)
            )
    except Exception:
        pass
    return None


# ── 执行 ──────────────────────────────────────────────────────────

def _run_single_task(
    task_id: str, task_text: str, task_type: str,
    backend: LlmBackend, backend_name: str, profiler_inst: LlmProfiler | None,
) -> TaskResult:
    orch = Orchestrator(
        backend=backend,
        profiler=profiler_inst,
        router_mode="quant",
        routing_memory=ProtocolMemory("runs/memory.json"),
    )
    trace = orch.solve(task_text)
    profile = trace.profile
    verification = trace.verification
    return TaskResult(
        task_id=task_id, task_type=task_type, backend=backend_name,
        task_text=task_text,
        complexity=profile.complexity,
        verifiability=profile.verifiability,
        topology_used=trace.topology.value,
        accepted=verification.accepted if verification else False,
        support_rate=verification.support_rate if verification else 0.0,
        claim=trace.final_answer or "",
        evidence_count=sum(1 for m in trace.messages if m.evidence),
    )


def _assert_fields(result: TaskResult) -> None:
    assert 0.0 <= result.complexity <= 1.0, f"complexity={result.complexity}"
    assert 0.0 <= result.verifiability <= 1.0, f"verifiability={result.verifiability}"
    assert result.topology_used in ("single_agent", "supervisor_worker", "review_loop"), (
        f"topology={result.topology_used}"
    )
    assert isinstance(result.accepted, bool)
    assert 0.0 <= result.support_rate <= 1.0, f"support_rate={result.support_rate}"
    assert isinstance(result.claim, str) and result.claim.strip(), "claim empty"
    if result.accepted:
        assert result.evidence_count >= 1, f"accepted but evidence={result.evidence_count}"


def _soft_expect(condition: bool, message: str) -> None:
    if not condition:
        warnings.warn(message, stacklevel=2)


# ── Parametrized test ─────────────────────────────────────────────

@pytest.mark.parametrize("task_id, task_text, task_type", TASKS)
def test_smoke_task(
    task_id: str, task_text: str, task_type: str,
    llm_backend: LlmBackend, backend_name: str,
    profiler: LlmProfiler | None, collector: ResultCollector,
) -> None:
    result = _run_single_task(
        task_id, task_text, task_type,
        backend=llm_backend, backend_name=backend_name, profiler_inst=profiler,
    )
    collector.add(result)

    _assert_fields(result)

    # 软断言
    if task_id == "T09":
        _soft_expect(result.topology_used == "single_agent",
                     f"T09 expected SINGLE_AGENT, got {result.topology_used}")
    if task_id == "T10":
        _soft_expect(result.topology_used == "review_loop",
                     f"T10 expected REVIEW_LOOP, got {result.topology_used}")
    if task_id in ("T07", "T08"):
        _soft_expect(result.verifiability >= 0.5,
                     f"{task_id} verifiability={result.verifiability:.3f} (<0.5)")
    if task_id in ("T01", "T02"):
        _soft_expect(result.complexity >= 0.4,
                     f"{task_id} complexity={result.complexity:.3f} (<0.4)")

    print("CSV:", collector.csv_line(result))


# ── 直接运行入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    args = [
        "scripts/smoke_test.py",
        "-v",
        "--backend", "deepseek",
        *sys.argv[1:],
    ]
    # 确保项目根目录是 cwd
    import os as _os
    _os.chdir(str(_PROJECT_ROOT))
    sys.exit(pytest.main(args))
