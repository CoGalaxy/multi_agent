"""
任务画像模块 — 输入自然语言任务，输出二维特征向量 (complexity, verifiability)。

提供两种实现：
- profile_task(): 关键词匹配规则（baseline），零延迟、可复现
- LlmProfiler: 用小模型做 few-shot 分类，准确率更高，
  作为论文中"从规则到学习"的演进对比。

实验设计：baseline (rules) vs 0.6B vs 4B 三组消融，分析分类器
参数量对整体协作质量的影响。
"""

from __future__ import annotations

import json
import logging
import re

from verifiable_multi_agent.contracts import TaskProfile

logger = logging.getLogger(__name__)

# ── 规则启发式关键词表 ──────────────────────────────────────────
# complexity: 任务有多难分解
#   - 多步骤标记 → 需要更多规划/分解
#   - 长任务/多从句 → 隐含更高分解成本
COMPLEXITY_HINTS = {
    # 中文 — 多步骤/复杂操作
    "比较", "对比", "分析", "评估", "设计", "实现", "构建", "部署",
    "优化", "重构", "集成", "配置", "规划", "协调", "分解",
    "并且", "同时", "然后", "之后", "以及",
    # 中文 — 需要多维度思考
    "多个", "多种", "复杂", "综合", "全面",
    # 中文 — 工程/执行类
    "生成", "执行", "调用", "开发", "编写", "计划",
    # 英文
    "compare", "comparison", "analyze", "design", "implement", "build",
    "deploy", "optimize", "refactor", "integrate", "configure",
    "and", "then", "after", "multiple", "complex", "comprehensive",
    "generate", "develop", "execute",
}

# verifiability: 输出有多难被验证
#   - 含主观/定性判断的任务 → 验证难（没有唯一正确答案）
#   - 含"比较""评估""建议"等 → 需要多维度交叉验证
#   - 含安全/风险/合规 → 输出正确性难以自动验证
#   - 含核查/验证/证伪 → 任务本身要求事实核查，verifiability 必然高
VERIFIABILITY_HINTS = {
    # 中文 — 需要验证/审查
    "验证", "检查", "审查", "审计", "核验", "确认", "测试",
    "核查", "是否正确", "反驳", "证伪", "事实核查",
    "确认来源", "给出证据",
    # 中文 — 主观/定性（难验证）
    "比较", "对比", "评估", "建议", "推荐", "分析", "判断",
    "研究", "探索", "调查", "设计", "优化",
    # 中文 — 不确定性标记
    "可能", "也许", "尝试", "最新", "当前",
    # 中文 — 安全/风险/合规（高验证难度）
    "安全", "风险", "合规", "约束", "隐私", "权限", "认证",
    "加密", "删除", "支付", "医疗", "法律", "生产",
    # 英文
    "verify", "validate", "check", "audit", "confirm", "test",
    "fact-check", "falsify", "verify the claim",
    "compare", "evaluate", "recommend", "analyze", "judge",
    "research", "investigate", "explore", "design", "optimize",
    "maybe", "perhaps", "latest", "current",
    "safety", "security", "risk", "compliance", "privacy",
    "delete", "payment", "medical", "legal", "production",
}


def profile_task(task: str) -> TaskProfile:
    """规则启发式 baseline — 零延迟，中英双语支持。

    complexity 评估：
    - 基于任务文本长度、从句数量和关键词命中
    - 短句（<20 字）且无关键词 → 低复杂度
    - 多从句/多关键词 → 高复杂度

    verifiability 评估：
    - 基于是否含"比较""验证""评估"等难以自动验证的关键词
    - 简单事实性问答 → 低 verifiability
    - 需要主观判断/多维度对比 → 高 verifiability
    """
    text = task.lower()

    # 从句拆分（中英文标点 + 连接词）
    clauses = re.split(r"[,;，；。、\n]| and | then | after | 然后 | 之后 | 以及 | 并且 ", text)
    clause_count = max(1, len([c for c in clauses if c.strip()]))

    # 关键词命中
    cn_hits = {h for h in COMPLEXITY_HINTS | VERIFIABILITY_HINTS if h in text}
    en_tokens = set(re.findall(r"[a-zA-Z_]+", text))
    all_hints = cn_hits | en_tokens

    complexity_hits = len(all_hints & COMPLEXITY_HINTS)
    verifiability_hits = len(all_hints & VERIFIABILITY_HINTS)

    # complexity: 从句数 + 关键词命中的加权
    #   clause_count 贡献最多 0.5（6 从句封顶）
    #   关键词命中贡献最多 0.5（4 个命中封顶）
    clause_score = min(clause_count / 6, 1.0) * 0.5
    keyword_score = min(complexity_hits / 4, 1.0) * 0.5
    # 长文本额外加分
    length_bonus = min(len(task) / 200, 1.0) * 0.2
    complexity = round(min(clause_score + keyword_score + length_bonus, 1.0), 3)

    # verifiability: 关键词命中 + 是否包含"比较/验证/评估"类强信号
    #   强信号词直接拉高 verifiability
    strong_signals = {"比较", "对比", "验证", "审查", "评估", "compare", "verify", "evaluate", "audit",
                      "安全", "风险", "合规", "安全约束", "隐私", "security", "safety", "risk", "compliance",
                      "核查", "是否正确", "反驳", "证伪", "事实核查", "fact-check", "falsify"}
    has_strong_signal = bool(all_hints & strong_signals)
    verifiability = round(min(verifiability_hits / 3, 1.0) if has_strong_signal else min(verifiability_hits / 5, 1.0), 3)

    return TaskProfile(task=task, complexity=complexity, verifiability=verifiability)


# ── LLM-based profiler prompt ───────────────────────────────────
_PROFILE_PROMPT = """Classify this task into two dimensions. Output ONLY a JSON object.
Classify both Chinese and English tasks equally well.

Dimensions:
- complexity (0-1): How hard is it to decompose this task into subtasks?
  Consider: number of distinct steps/clauses, need for multi-step planning,
  interdependence between parts, domain knowledge breadth.
  (任务有多难分解为子任务？考虑：独立步骤数量、是否需要多步规划、各部分间的依赖关系)

- verifiability (0-1): How hard is it to verify the output is correct?
  Consider: presence of subjective judgment, comparison/evaluation tasks,
  need for multi-dimensional cross-checking, absence of a single correct answer.
  (输出有多难验证其正确性？考虑：是否主观判断、是否比较/评估类任务、是否需要多维度交叉验证)

CRITICAL verifiability rules:
- If the task asks to check/verify/falsify/audit a claim → verifiability >= 0.6
  (如果任务要求核查/验证/证伪/审计某个说法 → verifiability >= 0.6)
- If the task asks to confirm whether something is true → verifiability >= 0.6
  (如果任务要求确认某事是否属实 → verifiability >= 0.6)
- If the task is purely describe/explain/summarize → verifiability may be low
  (如果任务只是描述/解释/总结 → verifiability 可以较低)
- If the task contains 核查/验证/是否正确/反驳/证伪 → verifiability >= 0.6
  (If task contains words like verify/check/audit/falsify in any language → verifiability >= 0.6)

Examples:
1. Say hello → {"complexity": 0.0, "verifiability": 0.0}
2. 打个招呼 → {"complexity": 0.0, "verifiability": 0.0}
3. 用三句话解释什么是二分查找 → {"complexity": 0.1, "verifiability": 0.15}
4. Compare AutoGen and CAMEL architectures → {"complexity": 0.55, "verifiability": 0.7}
5. 比较 AutoGen 和 CAMEL 的架构差异并给出适用场景 → {"complexity": 0.6, "verifiability": 0.75}
6. 删除生产数据库中的用户数据并验证备份 → {"complexity": 0.45, "verifiability": 0.85}
7. 请验证以下说法是否正确：RAG 可以消除幻觉 → {"complexity": 0.4, "verifiability": 0.85}
8. 请核查：MetaGPT 是否使用了 ContractMessage → {"complexity": 0.3, "verifiability": 0.8}
9. 设计一个复杂的多智能体协作系统 → {"complexity": 0.85, "verifiability": 0.7}
10. 用一句话总结什么是 Chain-of-Thought → {"complexity": 0.05, "verifiability": 0.05}

Now classify: "{task}"

JSON:"""


class LlmProfiler:
    """用小模型（0.6B-1.5B）做任务画像，替代关键词规则。

    如果 LLM 返回的 JSON 解析失败，自动 fallback 到 profile_task()。
    """

    def __init__(self, backend) -> None:
        self._backend = backend

    def profile(self, task: str) -> TaskProfile:
        prompt = _PROFILE_PROMPT.replace("{task}", task)
        try:
            raw = self._backend.complete(system="", user=prompt)
            data = json.loads(raw)
            return TaskProfile(
                task=task,
                complexity=float(data["complexity"]),
                verifiability=float(data["verifiability"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("LLM profiler failed (%s), falling back to keyword rules", exc)
            return profile_task(task)
