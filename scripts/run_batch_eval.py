from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VMA batch evaluation cases.")
    parser.add_argument("--cases", type=Path, default=Path("data/batch_eval_cases.jsonl"))
    parser.add_argument("--backend", default="deepseek")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/batch_eval_summary.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("runs/batch_eval_outputs"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    _load_env_file()

    cases = _load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.backend == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is not set. Set it before running DeepSeek batch evaluation.", file=sys.stderr)
        return 2

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} {case['category']}")
        command = [
            "vma",
            case["task"],
            "--backend",
            args.backend,
            "--router",
            "quant",
            "--contract-report",
            "--show-topology",
        ]
        if args.base_url:
            command.extend(["--base-url", args.base_url])
        if args.model:
            command.extend(["--model", args.model])

        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        output_path = args.raw_dir / f"{index:02d}_{case['id']}.txt"
        output_path.write_text(result.stdout, encoding="utf-8")
        if result.returncode != 0:
            output_path.with_suffix(".stderr.txt").write_text(result.stderr, encoding="utf-8")
            rows.append(_error_row(case, result.returncode, result.stderr, output_path))
            continue
        rows.append(_summary_row(case, result.stdout, output_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary: {args.output}")
    return 0


def _load_env_file(path: Path = Path(".env")) -> None:
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


def _load_cases(path: Path) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _summary_row(case: dict[str, str], output: str, output_path: Path) -> dict[str, Any]:
    accepted = _bool_or_text(_extract(r"accepted=(True|False)", output))
    return {
        "id": case["id"],
        "category": case["category"],
        "status": "ok" if accepted is True else "model_quality_failed",
        "output_path": str(output_path),
        "topology": _extract(r"selected=([A-Z_]+)", output).lower(),
        "nodes": _extract_generated_nodes(output),
        "max_review_loops": _int_or_text(_extract(r"max_review_loops=(\d+)", output)),
        "review_loops_used": _int_or_text(_extract(r"review_loops_used=(\d+)", output)),
        "support_rate": _float_or_text(_extract(r"support_rate=([0-9.]+)", output)),
        "task_coverage": _float_or_text(_extract(r"task_coverage=([0-9.]+)", output)),
        "accepted": accepted,
        "violations": _parse_violations(_extract(r"violations=(.+)", output)),
    }


def _error_row(case: dict[str, str], returncode: int, stderr: str, output_path: Path) -> dict[str, Any]:
    return {
        "id": case["id"],
        "category": case["category"],
        "status": _error_status(stderr),
        "returncode": returncode,
        "output_path": str(output_path),
        "stderr_path": str(output_path.with_suffix(".stderr.txt")),
        "topology": "",
        "nodes": "",
        "max_review_loops": "",
        "review_loops_used": "",
        "support_rate": "",
        "task_coverage": "",
        "accepted": False,
        "violations": [stderr.strip().replace("\n", " ")[:500]],
    }


def _extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _extract_generated_nodes(output: str) -> str:
    match = re.search(r"\[Generated Topology\]\s*\n(?P<body>.*?)(?:\nblocked=)", output, re.S)
    if not match:
        return ""
    lines = [line.strip() for line in match.group("body").splitlines() if line.strip()]
    return " ".join(lines)


def _int_or_text(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _float_or_text(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _bool_or_text(value: str) -> bool | str:
    if value == "True":
        return True
    if value == "False":
        return False
    return value


def _parse_violations(value: str) -> list[str]:
    if not value or value == "[]":
        return []
    if value.startswith("[") and value.endswith("]"):
        return [value]
    return [value]


def _error_status(stderr: str) -> str:
    text = stderr.lower()
    backend_markers = (
        "llm backend request failed",
        "connecterror",
        "ssl:",
        "httpx",
        "httpcore",
        "timeout",
    )
    if any(marker in text for marker in backend_markers):
        return "backend_error"
    return "runtime_error"


if __name__ == "__main__":
    raise SystemExit(main())
