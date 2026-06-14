#!/usr/bin/env python3
"""Profiler benchmark — compare keyword rules, 0.8b, and 4b profilers.

Uses the 4b model as the reference (strongest available classifier) and
reports how well the keyword-rules baseline and the 0.8b tiny model
approximate its decisions.

Output:
  1. Topology agreement matrix (rules vs 4b, 0.8b vs 4b)
  2. Dimension-level MAE for tool_need, uncertainty, step_count, risk
  3. Disagreement case listing for manual review
  4. Summary table suitable for thesis tables

Usage:
  python scripts/benchmark.py                           # full benchmark
  python scripts/benchmark.py --sample 20               # quick 20-task smoke test
  python scripts/benchmark.py --disagreements-only       # only show mismatches
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifiable_multi_agent.backends import OllamaBackend
from verifiable_multi_agent.profiler import LlmProfiler, profile_task
from verifiable_multi_agent.quantitative_router import QuantitativeRouter, infer_input_requirements, topology_from_spec

# ─── config ──────────────────────────────────────────────────
BENCHMARK_FILE = Path(__file__).resolve().parent.parent / "data" / "benchmark_tasks.json"
MODEL_SMALL = "qwen3.5:0.8b"
MODEL_LARGE = "qwen3.5:4b"

# ─── benchmark runner ────────────────────────────────────────

def load_tasks(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_profiler(name: str, profiler_fn, tasks: list[dict], verbose: bool = True) -> list[dict]:
    """Run a profiler against all tasks, return results with timing."""
    results = []
    total_start = time.perf_counter()
    for i, item in enumerate(tasks):
        task = item["task"]
        t0 = time.perf_counter()
        profile = profiler_fn(task)
        elapsed = time.perf_counter() - t0
        spec = QuantitativeRouter().route(profile, infer_input_requirements(task))
        topology = topology_from_spec(spec).value
        results.append({
            "task": task,
            "profile": profile,
            "topology": topology,
            "latency_ms": round(elapsed * 1000, 1),
        })
        if verbose:
            print(f"  [{name}] {i+1}/{len(tasks)}: {topology} ({elapsed*1000:.0f}ms)")
    total = time.perf_counter() - total_start
    if verbose:
        print(f"  [{name}] total: {total:.1f}s, avg: {total/len(tasks)*1000:.0f}ms/task\n")
    return results


def compare(reference: list[dict], candidate: list[dict], label: str) -> dict:
    """Compare candidate profiler results against reference."""
    n = len(reference)
    topo_agree = 0
    dim_errors = {"tool_need": [], "uncertainty": [], "step_count": [], "risk": []}
    disagreements = []

    for ref, cand in zip(reference, candidate):
        ref_p, cand_p = ref["profile"], cand["profile"]
        if ref["topology"] == cand["topology"]:
            topo_agree += 1
        else:
            disagreements.append({
                "task": ref["task"],
                "ref_topology": ref["topology"],
                "cand_topology": cand["topology"],
                "ref_complexity": ref_p.complexity,
                "cand_complexity": cand_p.complexity,
            })
        dim_errors["tool_need"].append(abs(ref_p.tool_need - cand_p.tool_need))
        dim_errors["uncertainty"].append(abs(ref_p.uncertainty - cand_p.uncertainty))
        dim_errors["step_count"].append(abs(ref_p.step_count - cand_p.step_count))
        dim_errors["risk"].append(abs(ref_p.risk - cand_p.risk))

    return {
        "label": label,
        "topology_accuracy": topo_agree / n,
        "topology_agree": topo_agree,
        "topology_total": n,
        "mae": {dim: sum(errs) / n for dim, errs in dim_errors.items()},
        "disagreements": disagreements,
    }


# ─── output formatting ───────────────────────────────────────

def print_report(
    tasks: list[dict],
    ref_results: list[dict],
    rules_results: list[dict],
    small_results: list[dict],
    rules_cmp: dict,
    small_cmp: dict,
) -> None:
    n = len(tasks)
    ref_latency = sum(r["latency_ms"] for r in ref_results) / n
    rules_latency = sum(r["latency_ms"] for r in rules_results) / n
    small_latency = sum(r["latency_ms"] for r in small_results) / n

    print("=" * 72)
    print("  Profiler Benchmark Report")
    print("=" * 72)
    print(f"  Tasks: {n}")
    print(f"  Reference model: {MODEL_LARGE}")
    print(f"  Average latency — rules: {rules_latency:.1f}ms | "
          f"0.8b: {small_latency:.1f}ms | 4b: {ref_latency:.1f}ms")
    print()

    # Topology agreement
    print("─" * 72)
    print("  Topology Agreement (vs 4b reference)")
    print("─" * 72)
    print(f"  {'Profiler':<20} {'Agreement':>10} {'Correct':>8} {'Total':>6}")
    print(f"  {'─'*20} {'─'*10} {'─'*8} {'─'*6}")
    for cmp in [rules_cmp, small_cmp]:
        print(f"  {cmp['label']:<20} {cmp['topology_accuracy']:>9.1%} "
              f"{cmp['topology_agree']:>7}  {cmp['topology_total']:>5}")
    print()

    # Dimension MAE
    print("─" * 72)
    print("  Mean Absolute Error per Dimension (vs 4b reference)")
    print("─" * 72)
    dims = ["tool_need", "uncertainty", "step_count", "risk"]
    header = f"  {'Profiler':<20}" + "".join(f"{d:>12}" for d in dims)
    print(header)
    print(f"  {'─'*20}" + "─" * 48)
    for cmp in [rules_cmp, small_cmp]:
        row = f"  {cmp['label']:<20}"
        for d in dims:
            row += f"  {cmp['mae'][d]:>9.3f}"
        print(row)
    print()

    # Disagreement analysis
    for cmp in [rules_cmp, small_cmp]:
        dis = cmp["disagreements"]
        print("─" * 72)
        print(f"  Disagreements: {cmp['label']} ({len(dis)}/{n})")
        print("─" * 72)
        if not dis:
            print("  (none — perfect agreement)")
            continue
        # Group by mismatch type
        overestimate = [d for d in dis
                        if _topo_rank(d["cand_topology"]) > _topo_rank(d["ref_topology"])]
        underestimate = [d for d in dis
                         if _topo_rank(d["cand_topology"]) < _topo_rank(d["ref_topology"])]
        print(f"  Overestimate (cand > ref): {len(overestimate)}")
        print(f"  Underestimate (cand < ref): {len(underestimate)}")
        print()
        for d in dis[:10]:  # show first 10
            arrow = "↑" if _topo_rank(d["cand_topology"]) > _topo_rank(d["ref_topology"]) else "↓"
            print(f"  {arrow} ref={d['ref_topology']:<20} cand={d['cand_topology']:<20}"
                  f"  [{d['ref_complexity']:.3f} vs {d['cand_complexity']:.3f}]")
            print(f"    {d['task'][:90]}...")
        if len(dis) > 10:
            print(f"  ... and {len(dis) - 10} more")
        print()

    # Topology distribution
    print("─" * 72)
    print("  Topology Distribution")
    print("─" * 72)
    for label, results in [("4b (ref)", ref_results), ("rules", rules_results), ("0.8b", small_results)]:
        dist = {}
        for r in results:
            dist[r["topology"]] = dist.get(r["topology"], 0) + 1
        parts = [f"{t}={c}" for t, c in sorted(dist.items())]
        print(f"  {label:<12} {', '.join(parts)}")
    print("=" * 72)


def _topo_rank(topo: str) -> int:
    return {"single_agent": 0, "supervisor_worker": 1, "review_loop": 2}[topo]


# ─── main ────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark profiler accuracy")
    parser.add_argument("--sample", type=int, default=0, help="Run on first N tasks only")
    parser.add_argument("--disagreements-only", action="store_true", help="Only show mismatches")
    args = parser.parse_args()

    tasks = load_tasks(BENCHMARK_FILE)
    if args.sample:
        tasks = tasks[:args.sample]
    print(f"Loaded {len(tasks)} tasks from {BENCHMARK_FILE}\n")

    # 1. Reference: 4b model
    print("=== Reference: 4b model ===")
    llm_large = OllamaBackend(model=MODEL_LARGE)
    large_profiler = LlmProfiler(llm_large)
    ref_results = run_profiler("4b", large_profiler.profile, tasks)

    # 2. Candidate: keyword rules (no LLM)
    print("=== Candidate: keyword rules ===")
    rules_results = run_profiler("rules", profile_task, tasks)

    # 3. Candidate: 0.8b model
    print("=== Candidate: 0.8b model ===")
    llm_small = OllamaBackend(model=MODEL_SMALL)
    small_profiler = LlmProfiler(llm_small)
    small_results = run_profiler("0.8b", small_profiler.profile, tasks)

    # Compute comparisons
    rules_cmp = compare(ref_results, rules_results, "keyword rules")
    small_cmp = compare(ref_results, small_results, "0.8b LLM")

    if args.disagreements_only:
        for cmp in [rules_cmp, small_cmp]:
            print(f"\n=== {cmp['label']} disagreements ({len(cmp['disagreements'])}) ===")
            for d in cmp["disagreements"]:
                arrow = "↑" if _topo_rank(d["cand_topology"]) > _topo_rank(d["ref_topology"]) else "↓"
                print(f"  {arrow} {d['task'][:100]}")
                print(f"    ref={d['ref_topology']} cand={d['cand_topology']}")
    else:
        print_report(tasks, ref_results, rules_results, small_results, rules_cmp, small_cmp)


if __name__ == "__main__":
    main()
