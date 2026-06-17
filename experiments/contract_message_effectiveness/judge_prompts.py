from __future__ import annotations

DELIVERABLE_AND_FACT_JUDGE_PROMPT = """你是实验评估器，需要判断一个模型回答是否存在事实幻觉或交付物幻觉。

任务：
{task}

必须覆盖的交付物：
{required_deliverables}

可依据的事实：
{reference_facts}

高风险幻觉点：
{hallucination_risks}

待评估回答：
{answer}

请只输出 JSON，不要输出 Markdown。字段如下：
{{
  "deliverable_complete": true or false,
  "missing_deliverables": ["缺失的交付物"],
  "fact_hallucination": true or false,
  "fact_hallucination_reasons": ["与参考事实冲突或无依据的具体表述"],
  "deliverable_hallucination": true or false,
  "deliverable_hallucination_reasons": ["声称完成但未实际给出的交付物"],
  "overall_quality_issue": true or false,
  "brief_reason": "一句话说明"
}}
"""


BASELINE_SYSTEM_PROMPT = """你是一个认真、谨慎的助手。请直接完成用户任务。
如果任务给出了材料或事实约束，只能依据这些材料作答，不要编造材料外事实。
如果任务要求代码、测试、证明、数据库结构或接口示例，必须给出实际交付物，不要只描述已经完成。
"""
