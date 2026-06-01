from __future__ import annotations

from verifiable_multi_agent.contracts import ContractMessage


GROUNDING_RULES = (
    "Ground every claim in the original task. "
    "Do not introduce a new topic that is absent from the task. "
    "Give a direct answer first whenever the task is answerable from the task wording or common domain knowledge. "
    "Do not over-refuse merely because no source document was attached. "
    "If important details are missing, answer at a safe general level first, then briefly state what would improve precision."
)


def system_prompt(agent_kind: str, task: str, instruction: str) -> str:
    return (
        f"You are a {agent_kind} agent. {instruction} "
        f"{language_rule(task)} "
        f"{GROUNDING_RULES}"
    )


def prompt_with_context(task: str, context: list[ContractMessage], extra: str | None = None) -> str:
    lines = [
        f"Original task: {task}",
        "Trace:",
    ]
    if not context:
        lines.append("- No prior messages.")
    for message in context:
        lines.append(
            f"- role={message.role.value}; subtask={message.subtask}; "
            f"claim={message.claim}; evidence={message.evidence}; action={message.action}"
        )
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def language_rule(task: str) -> str:
    if contains_cjk(task):
        return "The original task is in Chinese; respond entirely in Simplified Chinese, including headings and route labels."
    return "Respond in the same language as the original task."


def route_label(task: str) -> str:
    return "执行路线" if contains_cjk(task) else "Execution route"


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)
