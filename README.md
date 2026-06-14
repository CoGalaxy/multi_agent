# 可验证多智能体协同毕设原型

本项目是硕士毕设用的本地最小多智能体协同系统，不按工业化系统处理。项目主线只保留 **量化路由**：

```text
用户任务
-> TaskProfile
-> QuantitativeRouter
-> TopologySpec
-> CollaborationGraph
-> 多智能体协同
-> 真实 LLM API
-> 运行轨迹和最终结果
```

上面的英文名称是代码中的类名或字段名。在论文和说明文字中，可以对应表述为：

```text
TaskProfile        ：任务画像
QuantitativeRouter ：量化路由器
TopologySpec       ：拓扑规格
CollaborationGraph ：协作图
LLM API            ：大模型接口
运行轨迹字段 trace ：运行轨迹
```

## 三个研究内容

### 1. 结构化合约验证机制

入口文件：`src/verifiable_multi_agent/contract_verification.py`

核心数据结构：`ContractMessage`

每个智能体输出都必须被约束为一条合约消息，包含：

```text
claim    ：该智能体当前给出的主张或结论
evidence ：支撑主张的证据、上下文或上游消息
action   ：该智能体本轮实际采取的动作
```

验证结果记录为 `VerificationResult`：

```text
accepted     ：本轮协作是否通过验证
support_rate ：证据对主张的支撑比例
violations   ：缺失证据、缺失动作、空主张等问题
next_action  ：下一步动作
```

目前验证重点是结构化合约是否完整，以及证据与主张是否存在基本支撑关系。它不追求工业级审计，而是为了让毕设原型的协作过程可追踪、可检查、可展示。

### 2. 可量化路由驱动的受约束协作拓扑生成机制

量化路由入口文件：`src/verifiable_multi_agent/quantitative_routing.py`

拓扑生成入口文件：`src/verifiable_multi_agent/constrained_topology.py`

核心链路：

```text
TaskProfile
-> QuantitativeRouter
-> TopologySpec
-> ConstrainedTopologyGenerator
-> CollaborationGraph
-> GraphExecutor
```

对应中文含义：

```text
任务画像
-> 量化路由器
-> 拓扑规格
-> 受约束拓扑生成器
-> 协作图
-> 协作图执行器
```

任务画像使用两个维度描述任务：

```text
complexity    ：任务分解难度
verifiability ：结果验证难度
```

量化路由器是这一研究内容中的量化决策层。它不直接执行智能体，而是把任务画像和输入需求转换成拓扑规格：

```text
TaskProfile(complexity, verifiability)
+ InputRequirements
-> QuantitativeRouter
-> TopologySpec
```

输入需求用于补充任务文本中的显式要求：

```text
requires_material           ：是否需要给定材料
requires_external_query     ：是否需要外部查询
requires_database           ：是否涉及数据库查询描述
requires_destructive_action ：是否涉及破坏性操作
requires_verification       ：是否明确要求验证
requires_comparison         ：是否属于比较任务
```

拓扑规格记录量化路由后的能力需求：

```text
planning            ：是否需要规划
verification        ：是否需要验证
synthesis           ：是否需要最终合成
material_grounding  ：是否需要材料依据
tool_execution      ：是否涉及工具步骤描述
safety_review       ：是否需要安全审查
max_nodes           ：最大节点数
max_edges           ：最大边数
max_review_loops    ：最大审查循环次数
```

因此，研究内容 2 不是只做固定规则分支，而是先把任务转换为可解释的量化画像，再生成可验证的拓扑规格，最后由约束生成器落成实际协作图。

### 3. 历史路由记忆与失败率升级机制

入口文件：`src/verifiable_multi_agent/routing_memory.py`

核心数据结构：`ProtocolMemory`

每次运行后，系统可以记录：

```text
complexity    ：任务分解难度
verifiability ：结果验证难度
topology_used ：本次使用的拓扑
accepted      ：合约验证是否通过
support_rate  ：证据支撑比例
```

新任务到来时，系统在任务画像的二维空间里查找相似历史任务。如果相似任务失败率较高，就提升拓扑复杂度：

```text
SINGLE_AGENT -> SUPERVISOR_WORKER -> REVIEW_LOOP
```

这个机制对应毕设中的“根据历史回答复现成功路由，或根据历史失败率提高路由复杂度”。

## 三种拓扑结构

### 单智能体拓扑

代码枚举名：`SINGLE_AGENT`

适用任务：简单解释、简短总结、低复杂度且低验证难度任务。

图结构：

```text
Executor -> Verifier -> Synthesizer
```

中文含义：

```text
执行智能体 -> 验证智能体 -> 合成智能体
```

节点职责：

```text
Executor    ：直接生成候选答案
Verifier    ：做轻量合约验证
Synthesizer ：合成最终答案
```

特点：没有规划智能体，避免对简单任务过度拆解。

### 监督者-执行者拓扑

代码枚举名：`SUPERVISOR_WORKER`

适用任务：比较、分析、需要明确维度但风险不高的任务。

图结构：

```text
Planner -> Executor -> Verifier -> Synthesizer
```

中文含义：

```text
规划智能体 -> 执行智能体 -> 验证智能体 -> 合成智能体
```

节点职责：

```text
Planner     ：确定回答维度、比较标准或执行顺序
Executor    ：按规划生成候选答案
Verifier    ：检查候选答案是否覆盖规划要求
Synthesizer ：形成最终答案
```

示例：`比较 AutoGen 和 LangGraph 的架构差异` 会触发规划、验证、合成能力需求，因此选择 `SUPERVISOR_WORKER`。

### 审查循环拓扑

代码枚举名：`REVIEW_LOOP`

适用任务：验证难度高、风险高、需要审查或修订的任务。

基础图结构：

```text
Planner -> Executor -> SafetyVerifier -> Verifier -> Synthesizer
```

中文含义：

```text
规划智能体 -> 执行智能体 -> 安全验证智能体 -> 验证智能体 -> 合成智能体
```

当拓扑中包含修订节点，并且验证未通过时，可以进入审查循环：

```text
... -> Reviser -> Verifier -> ...
```

节点职责：

```text
Planner        ：先定义高风险任务的边界和检查点
Executor       ：生成初始候选答案
SafetyVerifier ：检查安全约束
Verifier       ：检查证据、覆盖度和一致性
Reviser        ：根据审查意见修订答案
Synthesizer    ：合成最终答案
```

特点：比监督者-执行者拓扑更强调验证、审查和必要时的修订。

## 后端

保留真实大模型接口：

| 后端 | 用途 |
| --- | --- |
| `vllm` | 本地模型对比主线 |
| `deepseek` | 云端模型对比主线 |
| `ollama` | 本地快速运行 |

不保留假后端运行模式。

## 运行示例

本地 vLLM：

```powershell
vma "比较 AutoGen 和 LangGraph 的架构差异" --backend vllm --base-url http://127.0.0.1:8000/v1 --router quant
```

DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
vma "比较 AutoGen 和 LangGraph 的架构差异" --backend deepseek --router quant
```

Ollama：

```powershell
vma "解释多智能体协同的优势" --backend ollama --model qwen3.5:4b --router quant
```

## 输出运行轨迹

```powershell
vma "task" --backend vllm --router quant --contract-report
vma "task" --backend vllm --router quant --json-trace
vma "task" --backend vllm --router quant --show-topology
vma "task" --backend vllm --router quant --save-run
```

常见运行轨迹字段：

```text
任务画像（Task Profile）
选择的拓扑（Topology）
量化路由细节（Quantitative Router）
生成后的协作图（Generated Topology）
实际执行过的节点（Graph Execution）
合约验证报告（Contract Report）
最终答案（Final Answer）
```

## 测试

```powershell
conda run -n vllm --no-capture-output python -m pytest -q
```
