"""LLMAgent 集成测试 — 覆盖正常解析和 fallback 两个路径。"""

import json
from unittest.mock import MagicMock

from verifiable_multi_agent.agents import LLMAgent
from verifiable_multi_agent.contracts import AgentRole, ContractMessage


def _mock_backend(response_text: str) -> MagicMock:
    """构造一个 mock backend，其 complete() 返回指定文本。"""
    backend = MagicMock()
    backend.is_real_llm = True
    backend.complete.return_value = response_text
    return backend


def _valid_json_response() -> str:
    return json.dumps(
        {
            "claim": "需要从架构、生态和性能三个维度进行比较。",
            "evidence": ["官方文档", "社区报告"],
            "action": "compare",
            "uncertainty": 0.3,
        },
        ensure_ascii=False,
    )


# ── 正常解析 ──────────────────────────────────────────────────────

def test_llm_agent_parses_valid_json() -> None:
    """LLMAgent 正确解析 backend 返回的有效 JSON。"""
    backend = _mock_backend(_valid_json_response())
    agent = LLMAgent(AgentRole.PLANNER, backend=backend)
    msg = agent.run(
        task="比较 AutoGen 和 CAMEL",
        context=[],
        subtask="定义比较维度",
        action="compare",
    )

    assert isinstance(msg, ContractMessage)
    assert msg.role == AgentRole.PLANNER
    assert msg.claim == "需要从架构、生态和性能三个维度进行比较。"
    assert msg.evidence == ["官方文档", "社区报告"]
    assert msg.action == "compare"
    assert msg.uncertainty == 0.3
    assert msg.subtask == "定义比较维度"


def test_llm_agent_accepts_minimal_args() -> None:
    """LLMAgent.run() 在只提供 task 和 context 时也能正常工作。"""
    backend = _mock_backend(
        json.dumps({"claim": "ok", "evidence": [], "action": "analyze", "uncertainty": 0.1})
    )
    agent = LLMAgent(AgentRole.EXECUTOR, backend=backend)
    msg = agent.run(task="hello", context=[])

    assert msg.claim == "ok"
    assert msg.uncertainty == 0.1


# ── Fallback ───────────────────────────────────────────────────────

def test_llm_agent_fallback_on_invalid_json() -> None:
    """backend 返回非法 JSON 时，LLMAgent fallback 为 parse error。"""
    backend = _mock_backend("这不是 JSON，是 LLM 的自由发挥。")
    agent = LLMAgent(AgentRole.VERIFIER, backend=backend)
    msg = agent.run(
        task="检查安全性",
        context=[],
        subtask="审计",
        action="verify",
    )

    assert msg.claim == "parse error"
    assert msg.evidence == []
    assert msg.uncertainty == 1.0
    # action 应使用传入值
    assert msg.action == "verify"


def test_llm_agent_fallback_on_empty_string() -> None:
    """backend 返回空字符串时 fallback。"""
    backend = _mock_backend("")
    agent = LLMAgent(AgentRole.SYNTHESIZER, backend=backend)
    msg = agent.run(task="test", context=[])

    assert msg.claim == "parse error"
    assert msg.uncertainty == 1.0


def test_llm_agent_fallback_on_missing_fields() -> None:
    """backend 返回合法 JSON 但缺少部分字段时，使用默认值填充。"""
    backend = _mock_backend('{"claim": "only claim"}')
    agent = LLMAgent(AgentRole.EXECUTOR, backend=backend)
    msg = agent.run(task="test", context=[])

    # claim 正常，其他字段使用默认值
    assert msg.claim == "only claim"
    assert msg.evidence == []
    assert msg.uncertainty == 0.5  # 默认值


# ── Prompt 构造 ───────────────────────────────────────────────────

def test_llm_agent_includes_context_in_prompt() -> None:
    """user prompt 应包含上下文消息的摘要。"""
    backend = _mock_backend(_valid_json_response())
    agent = LLMAgent(AgentRole.VERIFIER, backend=backend)
    context = [
        ContractMessage(
            role=AgentRole.EXECUTOR,
            subtask="draft",
            claim="AutoGen 更灵活。",
            evidence=["doc"],
            action="draft answer",
        )
    ]

    agent.run(task="compare frameworks", context=context, subtask="verify")

    # 验证 prompt 中包含上下文
    call_args = backend.complete.call_args
    user_prompt = call_args.kwargs["user"]
    assert "AutoGen 更灵活" in user_prompt
    assert "executor" in user_prompt.lower()


def test_llm_agent_system_prompt_contains_role() -> None:
    """system prompt 应包含 Agent 角色名。"""
    backend = _mock_backend(_valid_json_response())
    agent = LLMAgent(AgentRole.PLANNER, backend=backend)
    agent.run(task="test", context=[])

    call_args = backend.complete.call_args
    system_prompt = call_args.kwargs["system"]
    assert "planner" in system_prompt.lower()
    assert "JSON" in system_prompt
