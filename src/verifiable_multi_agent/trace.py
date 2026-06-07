from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from verifiable_multi_agent.contracts import AgentTrace
from verifiable_multi_agent.reporting import ContractReport, build_contract_report


class RunTrace(BaseModel):
    run_id: str
    created_at: str
    task: str
    task_profile: dict
    selected_topology: str
    topology_reason: str | None = None
    agents_executed: list[str]
    contract_messages: list[dict]
    verification_result: dict | None
    final_answer: str | None
    metrics: dict
    contract_report: ContractReport
    metadata: dict
    generated_topology: dict | None = None
    graph_execution: dict | None = None


def build_run_trace(trace: AgentTrace, run_id: str | None = None) -> RunTrace:
    report = build_contract_report(trace)
    generated_run_id = run_id or str(uuid4())
    return RunTrace(
        run_id=generated_run_id,
        created_at=datetime.now(UTC).isoformat(),
        task=trace.task,
        task_profile=trace.profile.model_dump(mode="json") | {"complexity": trace.profile.complexity},
        selected_topology=trace.topology.value,
        topology_reason=trace.topology_reason,
        agents_executed=[message.role.value for message in trace.messages],
        contract_messages=[message.model_dump(mode="json") for message in trace.messages],
        verification_result=trace.verification.model_dump(mode="json") if trace.verification else None,
        final_answer=trace.final_answer,
        metrics={
            "total_messages": report.total_messages,
            "supported_claims": report.supported_claims,
            "support_rate": report.support_rate,
            "evidence_coverage": report.evidence_coverage,
            "action_completeness": report.action_completeness,
            "complexity": trace.profile.complexity,
        },
        contract_report=report,
        metadata=trace.metadata,
        generated_topology=trace.metadata.get("generated_topology"),
        graph_execution=trace.metadata.get("graph_execution"),
    )


def run_trace_json(run_trace: RunTrace) -> str:
    return json.dumps(run_trace.model_dump(mode="json"), ensure_ascii=False, indent=2)


def save_run_trace(run_trace: RunTrace, runs_dir: Path = Path("runs")) -> Path:
    run_dir = runs_dir / run_trace.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "trace.json"
    output_path.write_text(run_trace_json(run_trace) + "\n", encoding="utf-8")
    return output_path
