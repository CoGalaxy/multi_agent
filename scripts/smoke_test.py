#!/usr/bin/env python3
"""冒烟测试 — 依次执行关键 CLI 路径，断言退出码和输出关键字。

用法:
  python scripts/smoke_test.py          # 运行步骤 1-3（mock）
  SMOKE_FULL=1 python scripts/smoke_test.py  # 运行全部 4 步（含 DeepSeek）

步骤 4 需要 DEEPSEEK_API_KEY 环境变量，CI 中未设置时自动跳过。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TASK = "比较 AutoGen 和 LangGraph 的架构差异并验证比较标准。"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"

# 尽量用已安装的 vma 命令，fallback 到模块调用
VMA = os.getenv("VMA_BIN", "vma")


def _run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    """运行 vma 命令，统一捕获输出。"""
    cmd = [VMA, *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=kwargs.pop("timeout", 60),  # type: ignore[arg-type]
        cwd=str(PROJECT_ROOT),
        **kwargs,  # type: ignore[arg-type]
    )


def _assert_ok(proc: subprocess.CompletedProcess[str], step: str) -> None:
    """断言退出码为 0。"""
    assert proc.returncode == 0, (
        f"[{step}] 退出码非 0: {proc.returncode}\n"
        f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )
    print(f"  [OK] [{step}] exit=0")


def _assert_contains(proc: subprocess.CompletedProcess[str], keyword: str, step: str) -> None:
    """断言 stdout 包含指定关键字。"""
    assert keyword in proc.stdout, (
        f"[{step}] 输出缺少关键字 '{keyword}'\nSTDOUT:\n{proc.stdout}"
    )


def _clean_runs() -> None:
    """清理上次运行的 runs/ 目录。"""
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)


# ── 步骤 1: mock + quant + contract-report ────────────────────────

def step1() -> None:
    """基本 contract-report 输出验证。"""
    _clean_runs()
    proc = _run([
        TASK,
        "--backend", "mock",
        "--router", "quant",
        "--contract-report",
    ])
    _assert_ok(proc, "step1")
    _assert_contains(proc, "[Contract Report]", "step1")
    _assert_contains(proc, "support_rate", "step1")
    print("  [OK] [step1] output contains [Contract Report] and support_rate")


# ── 步骤 2: mock + quant + json-trace + save-run ──────────────────

def step2() -> None:
    """JSON trace 输出并持久化到 runs/ 目录。"""
    _clean_runs()
    proc = _run([
        TASK,
        "--backend", "mock",
        "--router", "quant",
        "--json-trace",
        "--save-run",
    ])
    _assert_ok(proc, "step2")
    # JSON 输出应可解析
    data = json.loads(proc.stdout)
    assert data.get("run_id"), "JSON trace 缺少 run_id"

    # 断言 runs/ 目录下生成了 trace.json
    trace_files = list(RUNS_DIR.glob("*/trace.json"))
    assert len(trace_files) >= 1, f"runs/ 目录下未找到 trace.json: {list(RUNS_DIR.glob('*'))}"
    # 验证内容可解析
    trace_data = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace_data.get("task"), "trace.json 缺少 task 字段"
    print(f"  [OK] [step2] trace.json created under runs/ (run_id={data['run_id']})")


# ── 步骤 3: mock + quant + routing-memory + contract-report ───────

def step3() -> None:
    """启用 routing memory 后的 contract-report 和持久化验证。"""
    _clean_runs()
    memory_path = RUNS_DIR / "memory.json"
    proc = _run([
        TASK,
        "--backend", "mock",
        "--router", "quant",
        "--routing-memory",
        "--contract-report",
    ])
    _assert_ok(proc, "step3")
    # routing memory 修正可能在输出中出现
    # 断言 runs/memory.json 存在（由 ProtocolMemory 持久化）
    assert memory_path.exists(), f"runs/memory.json 未生成: {list(RUNS_DIR.glob('*')) if RUNS_DIR.exists() else 'runs/ 不存在'}"

    # 验证 memory.json 内容为有效 JSON 数组
    memory_data = json.loads(memory_path.read_text(encoding="utf-8"))
    assert isinstance(memory_data, list), "memory.json 不是 JSON 数组"
    assert len(memory_data) >= 1, "memory.json 为空数组（应至少有一条记录）"
    record = memory_data[0]
    for field in ("task_id", "complexity", "verifiability", "topology_used", "accepted", "support_rate"):
        assert field in record, f"memory.json 记录缺少字段: {field}"
    print(f"  [OK] [step3] runs/memory.json exists with {len(memory_data)} record(s)")


# ── 步骤 4: deepseek + quant + routing-memory + contract-report ───

def step4() -> None:
    """真实 DeepSeek API 调用（需要 DEEPSEEK_API_KEY）。"""
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("  [SKIP] [step4] DEEPSEEK_API_KEY not set")
        return

    _clean_runs()
    proc = _run([
        TASK,
        "--backend", "deepseek",
        "--router", "quant",
        "--routing-memory",
        "--contract-report",
    ], timeout=180)
    _assert_ok(proc, "step4")
    _assert_contains(proc, "claim", "step4")
    _assert_contains(proc, "accepted", "step4")
    print("  [OK] [step4] DeepSeek output contains claim and accepted")


# ── main ───────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Smoke Test — Verifiable Multi-Agent Scaffold")
    print(f"VMA_BIN={VMA}")
    print("=" * 60)

    steps = [step1, step2, step3]
    if os.getenv("SMOKE_FULL"):
        steps.append(step4)

    failed: list[str] = []
    for step_fn in steps:
        try:
            step_fn()
        except AssertionError as exc:
            failed.append(f"{step_fn.__name__}: {exc}")
            print(f"  [FAIL] {exc}")
        except Exception as exc:
            failed.append(f"{step_fn.__name__}: {exc}")
            print(f"  [ERROR] {exc}")

    print("=" * 60)
    if failed:
        print(f"SMOKE TEST FAILED — {len(failed)}/{len(steps)} 步骤失败:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("smoke test passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
