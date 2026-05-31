#!/usr/bin/env python3
"""End-to-end benchmark: multi-agent pipeline vs single-LLM baseline.

Evaluates the FULL pipeline (Profile → Route → Execute → Verify → Store),
not just the profiler/router. For each task:

  1. Run the multi-agent Orchestrator → final answer + trace
  2. Run a single LLM call (same model, no agents) → baseline answer
  3. Compare both against ground truth or via LLM judge

Metrics:
  - win_rate: how often multi-agent beats baseline
  - verification_impact: does Accepted=False correlate with worse answers?
  - escalation_benefit: does review_loop produce better answers?
  - cost: latency & token comparison

Usage:
  python scripts/benchmark_e2e.py                    # full run
  python scripts/benchmark_e2e.py --sample 10        # quick smoke test
  python scripts/benchmark_e2e.py --verbose           # show per-task details
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifiable_multi_agent.backends import OllamaBackend
from verifiable_multi_agent.orchestrator import Orchestrator
from verifiable_multi_agent.profiler import LlmProfiler, profile_task
from verifiable_multi_agent.router import select_topology

TASK_FILE = Path(__file__).resolve().parent.parent / "data" / "benchmark_e2e_tasks.json"
MODEL_MAIN = "qwen3.5:4b"
MODEL_PROFILE = "qwen3.5:0.8b"

# ─── judge prompt ────────────────────────────────────────────

_JUDGE_PROMPT = """You are an impartial judge. Compare two answers to the same task.

Task: {task}

Expected answer (ground truth or reference): {expected}

Answer A (multi-agent pipeline): {answer_a}

Answer B (single LLM call): {answer_b}

Which answer is better? Consider correctness, completeness, clarity, and whether it addresses the task.

Output ONLY a JSON object:
{{
  "winner": "A" or "B" or "tie",
  "score_a": 0.0-1.0,
  "score_b": 0.0-1.0,
  "reason": "one sentence explaining the decision"
}}

JSON:"""


# ─── objective evaluators ────────────────────────────────────

def evaluate_math(expected: str, answer: str) -> float:
    """Extract numbers and check if answer contains the expected result."""
    expected_nums = set(re.findall(r"\d+\.?\d*", expected))
    answer_nums = set(re.findall(r"\d+\.?\d*", answer))
    if not expected_nums:
        return 0.5  # can't evaluate
    overlap = len(expected_nums & answer_nums)
    return min(overlap / len(expected_nums), 1.0)


def evaluate_code(expected: str, answer: str) -> float:
    """Check if answer contains executable code matching key patterns in expected."""
    expected_patterns = set(re.findall(r"[a-zA-Z_]{3,}", expected))
    answer_patterns = set(re.findall(r"[a-zA-Z_]{3,}", answer))
    if not expected_patterns:
        return 0.5
    overlap = len(expected_patterns & answer_patterns)
    return min(overlap / len(expected_patterns), 1.0)


# ─── runners ─────────────────────────────────────────────────

def run_multi_agent(task: str, backend: OllamaBackend, profiler: LlmProfiler) -> dict:
    """Full multi-agent pipeline."""
    orch = Orchestrator(
        memory_path=Path("data/protocol_memory.jsonl"),
        backend=backend,
        profiler=profiler,
    )
    t0 = time.perf_counter()
    trace = orch.solve(task)
    elapsed = time.perf_counter() - t0

    topology = select_topology(trace.profile).value
    verification = trace.verification

    return {
        "answer": trace.final_answer or "",
        "topology": topology,
        "complexity": trace.profile.complexity,
        "accepted": verification.accepted if verification else None,
        "violations": verification.violations if verification else [],
        "message_count": len(trace.messages),
        "escalated": topology != trace.topology.value if hasattr(trace, 'topology') else False,
        "latency_ms": round(elapsed * 1000),
        "roles": [m.role.value for m in trace.messages],
    }


def run_baseline(task: str, backend: OllamaBackend) -> dict:
    """Single LLM call — no agents, no contracts, no verification."""
    t0 = time.perf_counter()
    answer = backend.complete(
        system="You are a helpful assistant. Answer the task directly and concisely.",
        user=task,
    )
    elapsed = time.perf_counter() - t0
    return {
        "answer": answer,
        "latency_ms": round(elapsed * 1000),
    }


def judge(task: str, expected: str, answer_a: str, answer_b: str, backend: OllamaBackend) -> dict:
    """LLM-as-judge comparing two answers against expected result."""
    prompt = _JUDGE_PROMPT.format(
        task=task, expected=expected,
        answer_a=answer_a[:1500], answer_b=answer_b[:1500],
    )
    try:
        raw = backend.complete(system="", user=prompt)
        return json.loads(raw)
    except (json.JSONDecodeError, KeyError):
        return {"winner": "tie", "score_a": 0.5, "score_b": 0.5, "reason": "judge parse error"}


# ─── main ────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks = json.loads(TASK_FILE.read_text())
    if args.sample:
        tasks = tasks[:args.sample]

    backend = OllamaBackend(model=MODEL_MAIN)
    profiler = LlmProfiler(OllamaBackend(model=MODEL_PROFILE))

    results = []
    wins_ma = 0   # multi-agent wins
    wins_bl = 0   # baseline wins
    ties = 0
    total_ma_lat = 0
    total_bl_lat = 0
    verified_reject = 0
    verified_total = 0

    for i, item in enumerate(tasks):
        task = item["task"]
        expected = item.get("expected", "")
        task_type = item.get("type", "subjective")

        print(f"[{i+1}/{len(tasks)}] {task_type}: {task[:80]}...")

        # Run both
        ma = run_multi_agent(task, backend, profiler)
        bl = run_baseline(task, backend)

        total_ma_lat += ma["latency_ms"]
        total_bl_lat += bl["latency_ms"]

        if ma["accepted"] is False:
            verified_reject += 1
        verified_total += 1

        # Evaluate
        if task_type in ("math", "factual"):
            score_ma = evaluate_math(expected, ma["answer"])
            score_bl = evaluate_math(expected, bl["answer"])
        elif task_type == "code":
            score_ma = evaluate_code(expected, ma["answer"])
            score_bl = evaluate_code(expected, bl["answer"])
        else:
            # subjective → LLM judge
            judge_result = judge(task, expected, ma["answer"], bl["answer"], backend)
            score_ma = judge_result.get("score_a", 0.5)
            score_bl = judge_result.get("score_b", 0.5)

        if score_ma > score_bl:
            winner = "multi_agent"
            wins_ma += 1
        elif score_bl > score_ma:
            winner = "baseline"
            wins_bl += 1
        else:
            winner = "tie"
            ties += 1

        results.append({
            "task": task[:100],
            "type": task_type,
            "winner": winner,
            "score_ma": round(score_ma, 3),
            "score_bl": round(score_bl, 3),
            "topology": ma["topology"],
            "complexity": ma["complexity"],
            "accepted": ma["accepted"],
            "violations": ma["violations"][:3],
            "ma_latency_ms": ma["latency_ms"],
            "bl_latency_ms": bl["latency_ms"],
            "ma_msg_count": ma["message_count"],
        })

        if args.verbose:
            v = ma["violations"][:2] if ma["violations"] else []
            print(f"  winner={winner}  ma_score={score_ma:.2f}  bl_score={score_bl:.2f}")
            print(f"  topology={ma['topology']}  complexity={ma['complexity']:.3f}  "
                  f"accepted={ma['accepted']}  violations={v}")
            print(f"  ma_latency={ma['latency_ms']}ms  bl_latency={bl['latency_ms']}ms")
            print(f"  ma_answer: {ma['answer'][:120]}...")
            print(f"  bl_answer: {bl['answer'][:120]}...")
            print()

    n = len(tasks)
    print("\n" + "=" * 66)
    print("  End-to-End Benchmark Report")
    print("=" * 66)
    print(f"  Tasks: {n}")
    print(f"  Model: {MODEL_MAIN}  |  Profiler: {MODEL_PROFILE}")
    print()
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'─'*35} {'─'*10}")
    print(f"  {'Multi-agent wins':<35} {wins_ma:>9}  ({wins_ma/n:.0%})")
    print(f"  {'Baseline (single LLM) wins':<35} {wins_bl:>9}  ({wins_bl/n:.0%})")
    print(f"  {'Ties':<35} {ties:>9}  ({ties/n:.0%})")
    print(f"  {'Avg multi-agent latency':<35} {total_ma_lat/n:>9.0f} ms")
    print(f"  {'Avg baseline latency':<35} {total_bl_lat/n:>9.0f} ms")
    print(f"  {'Latency ratio (ma/baseline)':<35} {total_ma_lat/total_bl_lat:>9.1f}x")
    print(f"  {'Verification reject rate':<35} {verified_reject:>8}/{verified_total}  ({verified_reject/verified_total:.0%})")
    print()

    # Breakdown by topology
    print(f"  {'─'*66}")
    print(f"  Breakdown by Topology")
    print(f"  {'─'*66}")
    topo_results = {}
    for r in results:
        t = r["topology"]
        if t not in topo_results:
            topo_results[t] = {"count": 0, "wins": 0, "score_sum": 0.0}
        topo_results[t]["count"] += 1
        topo_results[t]["wins"] += 1 if r["winner"] == "multi_agent" else 0
        topo_results[t]["score_sum"] += r["score_ma"]
    for topo in ["single_agent", "supervisor_worker", "review_loop"]:
        if topo in topo_results:
            tr = topo_results[topo]
            print(f"  {topo:<22} n={tr['count']:<3}  win_rate={tr['wins']/tr['count']:.0%}  "
                  f"avg_score={tr['score_sum']/tr['count']:.3f}")

    # Escalation analysis
    escalated = [r for r in results if r["topology"] == "review_loop"]
    if escalated:
        esc_wins = sum(1 for r in escalated if r["winner"] == "multi_agent")
        print(f"\n  review_loop tasks: {len(escalated)}, multi-agent win rate: {esc_wins/len(escalated):.0%}")

    # Verification analysis
    rejected = [r for r in results if r["accepted"] is False]
    accepted = [r for r in results if r["accepted"] is True]
    if rejected and accepted:
        rej_ma_score = sum(r["score_ma"] for r in rejected) / len(rejected)
        acc_ma_score = sum(r["score_ma"] for r in accepted) / len(accepted)
        print(f"\n  Verification impact:")
        print(f"    Rejected traces (n={len(rejected)}): avg score = {rej_ma_score:.3f}")
        print(f"    Accepted traces (n={len(accepted)}): avg score = {acc_ma_score:.3f}")
        print(f"    Note: rejected traces get synthesizer correction; "
              f"score reflects corrected output.")

    # Type breakdown
    print(f"\n  {'─'*66}")
    print(f"  Breakdown by Task Type")
    print(f"  {'─'*66}")
    type_results = {}
    for r in results:
        tt = r["type"]
        if tt not in type_results:
            type_results[tt] = {"count": 0, "ma_win": 0, "bl_win": 0}
        type_results[tt]["count"] += 1
        if r["winner"] == "multi_agent":
            type_results[tt]["ma_win"] += 1
        elif r["winner"] == "baseline":
            type_results[tt]["bl_win"] += 1
    for tt, tr in sorted(type_results.items()):
        print(f"  {tt:<15} n={tr['count']:<3}  ma_wins={tr['ma_win']}  "
              f"bl_wins={tr['bl_win']}  ties={tr['count']-tr['ma_win']-tr['bl_win']}")

    print("=" * 66)


if __name__ == "__main__":
    main()
