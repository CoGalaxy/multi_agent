# 可验证多智能体协同毕设原型

本项目是面向毕设的本地最小多智能体协同系统，不按工业化系统处理。主线保留**量化路由**，并围绕真实大模型接口运行：

```text
用户任务
-> 任务画像
-> 量化路由器
-> 拓扑规格
-> 受约束协作图
-> 多智能体协同
-> 真实大模型接口
-> 运行轨迹和最终结果
```

代码中的主要类名对应关系：

```text
TaskProfile              ：任务画像
QuantitativeRouter       ：量化路由器
InputRequirements        ：输入需求
CapabilityNeeds          ：能力需求
TopologySpec             ：拓扑规格
ConstrainedTopologyGenerator：受约束拓扑生成器
CollaborationGraph       ：协作图
GraphExecutor            ：协作图执行器
```

## 三个研究内容

### 1. 结构化合约验证机制

入口文件：`src/verifiable_multi_agent/contract_verification.py`

核心数据结构：`ContractMessage`、`VerificationResult`

每个智能体输出都被约束为结构化消息：

```text
主张    ：当前智能体给出的结论或中间判断
证据    ：支撑主张的上下文、上游消息或执行痕迹
动作    ：当前智能体本轮实际采取的动作
```

验证器检查消息是否有主张、证据和动作，并给出支持率、证据覆盖率、动作完整度和违规项。这个机制用于让协作过程可追踪、可检查、可展示。

### 2. 可量化路由驱动的受约束协作拓扑生成机制

量化路由入口文件：`src/verifiable_multi_agent/quantitative_router.py`

拓扑生成入口文件：`src/verifiable_multi_agent/topology/generator.py`

核心链路：

```text
任务画像
+ 输入需求
-> 量化路由器
-> 拓扑规格
-> 受约束拓扑生成器
-> 协作图
-> 协作图执行器
```

任务画像使用两个量化维度：

```text
任务复杂度      ：任务是否需要拆解、规划或多步协作
结果可验证性    ：结果是否需要更强验证、审查或安全检查
```

输入需求用于识别任务文本中的显式信号：

```text
是否需要给定材料
是否需要外部查询
是否涉及数据库
是否涉及破坏性操作
是否明确要求验证
是否属于比较任务
是否需要代码生成或测试
是否需要批判
是否需要修订
是否需要资料调研
是否需要安全审查
```

量化路由器把这些信号映射为能力需求：

```text
规划能力
材料依据能力
工具执行能力
代码测试能力
批判能力
修订能力
安全审查能力
验证能力
最终合成能力
```

拓扑生成采用**受约束动态生成**，不是完全自由图生成。系统只在有限智能体类型、固定顺序和预算约束内组合协作图。固定顺序为：

```text
规划者
-> 资料研究者
-> 工具执行者
-> 代码编写者/执行者
-> 测试者
-> 批判者
-> 修订者
-> 安全验证者
-> 验证者
-> 合成者
```

生成器只加入能力需求中需要的节点，但始终保证可执行图至少包含执行节点、验证节点和合成节点。若任务要求“根据给定材料”但没有提供材料，系统会生成阻塞图，不让模型凭空回答。

### 3. 历史路由记忆与失败率升级机制

入口文件：`src/verifiable_multi_agent/memory.py`

核心数据结构：`ProtocolMemory`

每次运行后，系统可以记录：

```text
任务复杂度
结果可验证性
本次使用的宏观拓扑等级
合约验证是否通过
证据支持率
```

新任务到来时，系统在任务画像空间中查找相似历史任务。如果相似任务失败率较高，就提高协作复杂度：

```text
单执行者级 -> 规划协作级 -> 审查修订级
```

## 三类拓扑的含义

三类拓扑在本项目中是**宏观协作等级**，不是固定图模板。实际执行结构以拓扑规格生成出的协作图为准。

### 单执行者级

代码枚举名：`SINGLE_AGENT`

典型结构：

```text
执行者 -> 验证者 -> 合成者
```

适合简单解释、短总结、低复杂度且低验证难度任务。

### 规划协作级

代码枚举名：`SUPERVISOR_WORKER`

典型结构：

```text
规划者 -> 执行者 -> 验证者 -> 合成者
```

适合比较、分析、需要明确回答维度或需要材料/工具辅助但风险不高的任务。

### 审查修订级

代码枚举名：`REVIEW_LOOP`

典型结构之一：

```text
规划者 -> 执行者 -> 安全验证者 -> 验证者 -> 合成者
```

当任务需要批判、修订、代码测试或高风险安全审查时，实际图中也可能出现：

```text
批判者 -> 修订者 -> 验证者
代码编写者 -> 测试者 -> 修订者 -> 验证者
```

## 动态拓扑示例

简单解释任务：

```text
执行者 -> 验证者 -> 合成者
```

结构化比较任务：

```text
规划者 -> 执行者 -> 验证者 -> 合成者
```

材料依赖任务：

```text
规划者 -> 资料研究者 -> 执行者 -> 验证者 -> 合成者
```

工具查询任务：

```text
规划者 -> 工具执行者 -> 执行者 -> 验证者 -> 合成者
```

代码测试任务：

```text
规划者 -> 代码编写者 -> 测试者 -> 修订者 -> 验证者 -> 合成者
```

高风险审查任务：

```text
规划者 -> 执行者 -> 安全验证者 -> 验证者 -> 合成者
```

批判修订任务：

```text
规划者 -> 执行者 -> 批判者 -> 修订者 -> 验证者 -> 合成者
```

## 后端

保留真实大模型接口：

| 后端 | 用途 |
| --- | --- |
| `vllm` | 本地模型对比主线 |
| `deepseek` | 云端模型对比主线 |
| `ollama` | 本地快速运行 |

不保留假后端运行模式。测试中的假后端只用于单元测试隔离。

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

## 运行轨迹

```powershell
vma "task" --backend vllm --router quant --contract-report
vma "task" --backend vllm --router quant --json-trace
vma "task" --backend vllm --router quant --show-topology
vma "task" --backend vllm --router quant --save-run
```

常见输出字段：

```text
任务画像
选择的宏观拓扑等级
量化路由细节
生成后的协作图
实际执行过的节点
合约验证报告
最终答案
```

## 测试

```powershell
conda run -n vllm --no-capture-output python -m pytest -q
```
