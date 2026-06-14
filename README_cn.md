# 可验证多智能体协同毕设原型

本项目是硕士毕设用的本地最小多智能体协同系统，不按工业化系统处理。项目主线只保留 **量化路由**。

代码链路如下：

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

中文表述如下：

```text
用户任务
-> 任务画像
-> 量化路由器
-> 拓扑规格
-> 协作图
-> 多智能体协同
-> 真实大模型接口
-> 运行轨迹和最终结果
```

## 三个研究内容

### 1. 结构化合约验证机制

入口文件：`src/verifiable_multi_agent/contract_verification.py`

核心数据结构：`ContractMessage`

每个智能体输出都必须包含：

```text
claim    ：当前主张或结论
evidence ：支撑主张的证据、上下文或上游消息
action   ：本轮实际采取的动作
```

验证结果记录为 `VerificationResult`：

```text
accepted     ：本轮协作是否通过验证
support_rate ：证据对主张的支撑比例
violations   ：缺失证据、缺失动作、空主张等问题
next_action  ：下一步动作
```

### 2. 可量化路由驱动的受约束协作拓扑生成机制

量化路由入口文件：`src/verifiable_multi_agent/quantitative_routing.py`

拓扑生成入口文件：`src/verifiable_multi_agent/constrained_topology.py`

代码链路：

```text
TaskProfile
-> QuantitativeRouter
-> TopologySpec
-> ConstrainedTopologyGenerator
-> CollaborationGraph
-> GraphExecutor
```

中文表述：

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

量化路由器把任务画像和输入需求转换成拓扑规格：

```text
TaskProfile(complexity, verifiability)
+ InputRequirements
-> QuantitativeRouter
-> TopologySpec
```

输入需求记录任务文本中的显式需求：

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

### 3. 历史路由记忆与失败率升级机制

入口文件：`src/verifiable_multi_agent/routing_memory.py`

核心数据结构：`ProtocolMemory`

历史记录包括：

```text
complexity    ：任务分解难度
verifiability ：结果验证难度
topology_used ：本次使用的拓扑
accepted      ：合约验证是否通过
support_rate  ：证据支撑比例
```

当相似历史任务失败率较高时，提高拓扑复杂度：

```text
SINGLE_AGENT -> SUPERVISOR_WORKER -> REVIEW_LOOP
```

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

存在修订节点且验证未通过时，可以进入：

```text
Reviser -> Verifier
```

中文含义：

```text
修订智能体 -> 验证智能体
```

## 后端

保留真实大模型接口：

| 后端 | 用途 |
| --- | --- |
| `vllm` | 本地模型对比主线 |
| `deepseek` | 云端模型对比主线 |
| `ollama` | 本地快速运行 |

不保留假后端运行模式。

## 运行示例

```powershell
vma "比较 AutoGen 和 LangGraph 的架构差异" --backend vllm --base-url http://127.0.0.1:8000/v1 --router quant
```

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
vma "比较 AutoGen 和 LangGraph 的架构差异" --backend deepseek --router quant
```

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

## 测试

```powershell
conda run -n vllm --no-capture-output python -m pytest -q
```
