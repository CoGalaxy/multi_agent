from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verifiable_multi_agent.backends import DeepSeekBackend
from verifiable_multi_agent.cli import load_env_file

from judge_prompts import DELIVERABLE_AND_FACT_JUDGE_PROMPT


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate contract message effectiveness experiment.")
    parser.add_argument("--raw", type=Path, default=Path(__file__).with_name("outputs") / "raw_runs.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs") / "evaluated_runs.jsonl")
    parser.add_argument("--summary", type=Path, default=Path(__file__).with_name("outputs") / "summary.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-llm-judge", action="store_true")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    judge_backend = None
    if not args.no_llm_judge:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("DEEPSEEK_API_KEY is not set; use --no-llm-judge or set the key.", file=sys.stderr)
            return 2
        judge_backend = DeepSeekBackend(
            api_key=api_key,
            model=args.model or os.getenv("DEEPSEEK_JUDGE_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )

    records = load_jsonl(args.raw)
    evaluated = evaluate_records(records, judge_backend=judge_backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, evaluated)
    summary = summarize(evaluated)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote evaluated runs: {args.output}")
    print(f"Wrote summary: {args.summary}")
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def evaluate_records(records: list[dict[str, Any]], judge_backend=None) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for row in records:
        if row.get("status") != "ok":
            evaluated.append({**row, "evaluation_status": "skipped_error"})
            continue
        judgment = judge_answer(row, judge_backend)
        contract_detected = hallucination_detected_by_contract(row, judgment)
        evaluated.append(
            {
                **row,
                "evaluation_status": "ok",
                "deliverable_complete": judgment["deliverable_complete"],
                "missing_deliverables": judgment["missing_deliverables"],
                "fact_hallucination": judgment["fact_hallucination"],
                "fact_hallucination_reasons": judgment["fact_hallucination_reasons"],
                "deliverable_hallucination": judgment["deliverable_hallucination"],
                "deliverable_hallucination_reasons": judgment["deliverable_hallucination_reasons"],
                "overall_quality_issue": judgment["overall_quality_issue"],
                "judge_reason": judgment["brief_reason"],
                "hallucination_detected_by_contract": contract_detected,
                "manual_review_required": judgment.get("judge_fallback", False),
            }
        )
    return evaluated


def judge_answer(row: dict[str, Any], judge_backend=None) -> dict[str, Any]:
    if judge_backend is not None:
        prompt = DELIVERABLE_AND_FACT_JUDGE_PROMPT.format(
            task=row["task"],
            required_deliverables=json.dumps(row.get("required_deliverables", []), ensure_ascii=False),
            reference_facts=json.dumps(row.get("reference_facts", []), ensure_ascii=False),
            hallucination_risks=json.dumps(row.get("hallucination_risks", []), ensure_ascii=False),
            answer=row.get("answer", ""),
        )
        try:
            raw = judge_backend.complete(system="", user=prompt, role="judge")
            data = parse_json_object(raw)
            return normalize_judgment(data)
        except Exception as exc:  # noqa: BLE001 - fallback keeps experiment usable.
            fallback = heuristic_judgment(row)
            fallback["brief_reason"] = f"judge fallback: {exc}; {fallback['brief_reason']}"
            fallback["judge_fallback"] = True
            return fallback
    fallback = heuristic_judgment(row)
    fallback["judge_fallback"] = True
    return fallback


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def normalize_judgment(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "deliverable_complete": bool(data.get("deliverable_complete", False)),
        "missing_deliverables": listify(data.get("missing_deliverables", [])),
        "fact_hallucination": bool(data.get("fact_hallucination", False)),
        "fact_hallucination_reasons": listify(data.get("fact_hallucination_reasons", [])),
        "deliverable_hallucination": bool(data.get("deliverable_hallucination", False)),
        "deliverable_hallucination_reasons": listify(data.get("deliverable_hallucination_reasons", [])),
        "overall_quality_issue": bool(data.get("overall_quality_issue", False)),
        "brief_reason": str(data.get("brief_reason", "")),
        "judge_fallback": False,
    }


def listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def heuristic_judgment(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer", "")
    missing = [
        item
        for item in row.get("required_deliverables", [])
        if not deliverable_hint_present(answer, str(item))
    ]
    deliverable_hallucination = bool(missing and claims_completion(answer))
    fact_reasons = [
        risk for risk in row.get("hallucination_risks", [])
        if risk_text_present(answer, str(risk))
    ]
    fact_hallucination = bool(fact_reasons)
    return {
        "deliverable_complete": not missing,
        "missing_deliverables": missing,
        "fact_hallucination": fact_hallucination,
        "fact_hallucination_reasons": fact_reasons,
        "deliverable_hallucination": deliverable_hallucination,
        "deliverable_hallucination_reasons": missing if deliverable_hallucination else [],
        "overall_quality_issue": bool(missing or fact_hallucination or deliverable_hallucination),
        "brief_reason": "heuristic evaluation",
    }


def deliverable_hint_present(answer: str, deliverable: str) -> bool:
    text = answer.lower()
    deliverable_lower = deliverable.lower()
    direct_terms = [term for term in re.findall(r"[a-zA-Z_]{3,}", deliverable_lower) if term not in {"python"}]
    if direct_terms and any(term in text for term in direct_terms):
        return True
    cjk_terms = [deliverable[index:index + 2] for index in range(len(deliverable) - 1)]
    hits = sum(1 for term in cjk_terms if "\u4e00" <= term[0] <= "\u9fff" and term in answer)
    return hits >= 2


def claims_completion(answer: str) -> bool:
    markers = (
        "已完成", "完整实现", "测试通过", "验证通过", "证明完成", "均已覆盖",
        "complete", "implemented", "all tests pass", "verified",
    )
    lower = answer.lower()
    return any(marker.lower() in lower for marker in markers)


def risk_text_present(answer: str, risk: str) -> bool:
    # The risk text is descriptive, so only use it as a conservative fallback.
    tokens = [token for token in re.findall(r"[a-zA-Z_]{4,}", risk.lower()) if token not in {"claim", "test"}]
    return bool(tokens and all(token in answer.lower() for token in tokens[:2]))


def hallucination_detected_by_contract(row: dict[str, Any], judgment: dict[str, Any]) -> bool:
    if row.get("group") != "contract":
        return False
    has_hallucination = judgment["fact_hallucination"] or judgment["deliverable_hallucination"]
    if not has_hallucination:
        return False
    violations = row.get("violations") or []
    return row.get("accepted") is False or bool(violations)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("evaluation_status") == "ok"]
    by_group = {group: summarize_group([row for row in ok_rows if row.get("group") == group]) for group in ("baseline", "contract")}
    baseline_h = by_group["baseline"].get("hallucination_rate", 0.0)
    contract_h = by_group["contract"].get("hallucination_rate", 0.0)
    return {
        "total_rows": len(rows),
        "evaluated_rows": len(ok_rows),
        "groups": by_group,
        "hallucination_reduction": round(baseline_h - contract_h, 3),
        "contract_detection_rate": by_group["contract"].get("hallucination_detection_rate", 0.0),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "hallucination_rate": 0.0,
            "fact_hallucination_rate": 0.0,
            "deliverable_hallucination_rate": 0.0,
            "deliverable_complete_rate": 0.0,
            "accepted_rate": 0.0,
            "average_task_coverage": None,
            "hallucination_detection_rate": 0.0,
        }
    hallucinated = [row for row in rows if row.get("fact_hallucination") or row.get("deliverable_hallucination")]
    task_coverages = [row.get("task_coverage") for row in rows if isinstance(row.get("task_coverage"), (int, float))]
    detected = [row for row in hallucinated if row.get("hallucination_detected_by_contract")]
    return {
        "count": len(rows),
        "hallucination_rate": rate(len(hallucinated), len(rows)),
        "fact_hallucination_rate": rate(sum(1 for row in rows if row.get("fact_hallucination")), len(rows)),
        "deliverable_hallucination_rate": rate(sum(1 for row in rows if row.get("deliverable_hallucination")), len(rows)),
        "deliverable_complete_rate": rate(sum(1 for row in rows if row.get("deliverable_complete")), len(rows)),
        "accepted_rate": rate(sum(1 for row in rows if row.get("accepted") is True), len(rows)),
        "average_task_coverage": round(sum(task_coverages) / len(task_coverages), 3) if task_coverages else None,
        "hallucination_detection_rate": rate(len(detected), len(hallucinated)) if hallucinated else 0.0,
    }


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
