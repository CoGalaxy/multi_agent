from verifiable_multi_agent.contract import ContractVerifier
from verifiable_multi_agent.contracts import AgentRole, ContractMessage


def test_contract_verifier_detects_claim_evidence_mismatch() -> None:
    message = ContractMessage(
        role=AgentRole.EXECUTOR,
        subtask="check result",
        claim="结果正确，无错误",
        evidence=["结果正确，通过检查", "检查通过且结果正确", "执行结果包含 error"],
        action="report result",
    )

    result = ContractVerifier().verify([message])

    assert result.accepted
    assert result.support_rate < 1.0
    assert any("soft_claim_evidence_mismatch" in violation for violation in result.violations)


def test_contract_verifier_does_not_misclassify_proof_verification_as_code_test() -> None:
    task = (
        "证明：对任意正整数 n，sum k^3 = [n(n+1)/2]^2。"
        "要求验证 n=1,n=2,n=3 三个基础情况，并写出归纳步骤。"
    )
    message = ContractMessage(
        role=AgentRole.SYNTHESIZER,
        subtask="final",
        claim="假设 P(n) 成立，因此 P(n+1) 成立，基础情况 n=1,n=2,n=3 均成立，所以得证。",
        evidence=["证明步骤包含假设、因此、所以、得证。"],
        action="synthesize",
    )

    result = ContractVerifier().verify([message], task=task)

    assert "missing_task_coverage:code" not in result.violations
    assert "missing_task_coverage:test" not in result.violations


def test_contract_verifier_rejection_lowers_support_rate() -> None:
    message = ContractMessage(
        role=AgentRole.VERIFIER,
        subtask="verify",
        claim="REJECTED: output claims tests exist but no concrete test evidence is provided",
        evidence=["output claims tests exist but no concrete test evidence is provided"],
        action="verify",
    )

    result = ContractVerifier().verify([message])

    assert not result.accepted
    assert result.support_rate < 1.0
    assert any("verifier_rejected" in violation for violation in result.violations)


def test_contract_verifier_reports_task_coverage_for_missing_answer_parts() -> None:
    task = "请设计数据库 schema，再写用户注册接口，最后验证接口是否覆盖两种登录方式。"
    message = ContractMessage(
        role=AgentRole.SYNTHESIZER,
        subtask="final",
        claim="方案支持 SSO 和本地账号两种登录方式，整体设计合理。",
        evidence=["上游消息提到了 SSO 和本地账号"],
        action="synthesize",
    )

    result = ContractVerifier().verify([message], task=task)

    assert not result.accepted
    assert result.task_coverage is not None
    assert result.task_coverage < 0.75
    assert any("missing_task_coverage" in violation for violation in result.violations)


def test_contract_verifier_accepts_answer_with_required_task_parts() -> None:
    task = "请设计数据库 schema，再写用户注册接口，最后验证接口是否覆盖两种登录方式。"
    message = ContractMessage(
        role=AgentRole.SYNTHESIZER,
        subtask="final",
        claim=(
            "Schema: user table contains id, email, password_hash, sso_provider, sso_id constraints. "
            "API: POST /register reads request fields and returns response. "
            "Tests: test_local_register and test_sso_register use assert to verify both login modes."
        ),
        evidence=[
            "schema table field constraint",
            "POST /register request response",
            "test_local_register assert; test_sso_register assert",
        ],
        action="synthesize",
    )

    result = ContractVerifier().verify([message], task=task)

    assert result.accepted
    assert result.task_coverage == 1.0


def test_task_coverage_requires_concrete_code_and_assertions() -> None:
    task = "写一个 Python 函数实现快速排序，并给出测试用例"
    message = ContractMessage(
        role=AgentRole.SYNTHESIZER,
        subtask="final",
        claim="提供完整的Python快速排序函数实现与配套测试用例，测试覆盖空列表和重复元素。",
        evidence=["上游说明实现了 quick_sort 并覆盖测试场景"],
        action="synthesize",
    )

    result = ContractVerifier().verify([message], task=task)

    assert not result.accepted
    assert result.task_coverage is not None
    assert result.task_coverage < 1.0
    assert "missing_task_coverage:code" in result.violations
    assert "missing_task_coverage:test" in result.violations


def test_task_coverage_requires_concrete_proof_steps() -> None:
    task = "证明两个偶数之和仍然是偶数"
    message = ContractMessage(
        role=AgentRole.SYNTHESIZER,
        subtask="final",
        claim="已经完成证明，结论成立。",
        evidence=["上游说明结论成立"],
        action="synthesize",
    )

    result = ContractVerifier().verify([message], task=task)

    assert not result.accepted
    assert "missing_task_coverage:proof" in result.violations
