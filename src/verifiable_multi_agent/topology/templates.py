from __future__ import annotations

from verifiable_multi_agent.topology.graph import AgentType


DEFAULT_ORDER = [
    AgentType.PLANNER,
    AgentType.RESEARCHER,
    AgentType.TOOL_EXECUTOR,
    AgentType.EXECUTOR,
    AgentType.CRITIC,
    AgentType.REVISER,
    AgentType.SAFETY_VERIFIER,
    AgentType.VERIFIER,
    AgentType.SYNTHESIZER,
]

CODE_TESTING_ORDER = [
    AgentType.PLANNER,
    AgentType.CODER,
    AgentType.TESTER,
    AgentType.REVISER,
    AgentType.VERIFIER,
    AgentType.SYNTHESIZER,
]


NODE_LABELS = {
    AgentType.PLANNER: "Planner",
    AgentType.RESEARCHER: "Researcher",
    AgentType.TOOL_EXECUTOR: "ToolExecutor",
    AgentType.EXECUTOR: "Executor",
    AgentType.CODER: "Coder",
    AgentType.TESTER: "Tester",
    AgentType.CRITIC: "Critic",
    AgentType.REVISER: "Reviser",
    AgentType.SAFETY_VERIFIER: "SafetyVerifier",
    AgentType.VERIFIER: "Verifier",
    AgentType.SYNTHESIZER: "Synthesizer",
}
