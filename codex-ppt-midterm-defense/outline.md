# 可验证多 Agent 协同架构原型中期答辩 PPT 大纲

## Slide 1: 可验证多 Agent 协同架构原型
- 关键点：
  - 课题方向：面向复杂长程任务的可验证 LLM Agent 与多 Agent 协同架构研究。
  - 当前阶段完成可运行原型。
  - 原型包含 CLI、合约消息、拓扑路由、图执行、运行报告与测试基线。
  - 中期答辩聚焦系统设计、实现进展和后续实验计划。
- 视觉想法：深色科技封面，中心标题，底部放阶段成果数字条。
- 页面角色：封面。
- 必需源图片：无。

## Slide 2: 复杂长程任务的协作问题
- 关键点：
  - 复杂任务通常包含多步骤、多角色、验证和工具调用。
  - 单次回答难以展示中间过程。
  - 多 Agent 协作需要记录每一步依据。
  - 系统需要支持过程追踪和过程验证。
- 视觉想法：三个问题模块，分别表示任务链路、角色协作、过程验证。
- 页面角色：背景与问题。
- 必需源图片：无。

## Slide 3: 本课题的研究目标
- 关键点：
  - 建立结构化的 Agent 间通信机制。
  - 根据任务特征选择协作拓扑。
  - 记录可复用的运行轨迹和协议记忆。
  - 用合约验证评估协作过程。
- 视觉想法：目标矩阵或三层结构图，避免口号式排比。
- 页面角色：研究目标。
- 必需源图片：无。

## Slide 4: 系统总体架构
- 关键点：
  - 输入任务先进入 Task Profiler。
  - Router 根据任务画像选择执行结构。
  - Orchestrator 或 GraphExecutor 组织 Agent 执行。
  - ContractMessage 记录协作过程。
  - Verifier 输出合约验证结果。
  - Trace 和 Memory 保留运行记录。
- 视觉想法：横向架构流程图，突出 `Task -> Profiler -> Router -> Executor -> Contracts -> Verifier -> Trace/Memory`。
- 页面角色：总体架构。
- 必需源图片：无。

## Slide 5: 合约消息机制
- 关键点：
  - `ContractMessage` 是 Agent 间通信的基本单元。
  - 字段包括 `role`、`subtask`、`claim`、`evidence`、`action`、`uncertainty`。
  - 验证器检查 claim 支撑、证据覆盖和动作完整性。
  - 合约消息为后续 trace 和报告提供数据基础。
- 视觉想法：数据结构剖面图，左侧字段，右侧验证指标。
- 页面角色：核心机制。
- 必需源图片：无。

## Slide 6: 任务画像与基础拓扑路由
- 关键点：
  - `TaskProfile` 使用 `tool_need`、`uncertainty`、`step_count`、`risk` 表示任务特征。
  - 基础拓扑包括 `SINGLE_AGENT`、`SUPERVISOR_WORKER`、`REVIEW_LOOP`。
  - 简单任务走直接执行和验证。
  - 中等复杂任务加入 Planner。
  - 高风险任务进入执行和验证循环。
- 视觉想法：四维任务画像面板，加三条拓扑路径。
- 页面角色：路由设计。
- 必需源图片：无。

## Slide 7: 编排器执行流程
- 关键点：
  - `Orchestrator.solve()` 串联画像、路由、执行、验证和存储。
  - 协议记忆用于复用相似任务的协作模式。
  - 预算机制限制运行步数。
  - 验证失败时可升级到 `REVIEW_LOOP`。
  - Synthesizer 输出最终答案。
- 视觉想法：主控流程泳道图，展示 Planner、Executor、Verifier、Synthesizer 的调用顺序。
- 页面角色：执行流程。
- 必需源图片：无。

## Slide 8: 定量 Router 设计
- 关键点：
  - `QuantitativeRouter` 基于复杂度特征生成 `TopologySpec`。
  - TCI 与 horizon、dependency_depth、tool_burden、evidence_burden、uncertainty、risk 等因素相关。
  - `CapabilityNeeds` 描述 planning、tool_execution、material_grounding、verification、safety_review、synthesis 等需求。
  - 缺少给定材料时返回 blocked 状态和 block_reason。
- 视觉想法：TCI 相关因素关系图，加能力需求开关矩阵。
- 页面角色：定量路由。
- 必需源图片：无。

## Slide 9: 协作图生成与执行
- 关键点：
  - Quant 路径使用 `TopologySpec -> CollaborationGraph -> GraphExecutor`。
  - `ConstrainedTopologyGenerator` 根据能力需求选择节点。
  - `GraphExecutor` 按顺序 DAG 执行节点。
  - 运行记录包含 executed_nodes、skipped_nodes、review_loops_used 和 execution_mode。
  - Review loop 受 max_review_loops 约束。
- 视觉想法：图生成与执行管线，配一段示例拓扑输出。
- 页面角色：图执行。
- 必需源图片：无。

## Slide 10: CLI 运行报告
- 关键点：
  - CLI 提供 `--contract-report`、`--json-trace`、`--save-run`、`--show-topology`。
  - Contract report 展示任务画像、拓扑、消息数量、support_rate 和 accepted 状态。
  - JSON trace 保留完整运行记录。
  - Save-run 将 trace 写入运行目录。
- 视觉想法：终端输出风格页面，展示报告结构和关键字段。
- 页面角色：运行示例。
- 必需源图片：无。

## Slide 11: 当前实现与测试结果
- 关键点：
  - 主要模块包括 contracts、orchestrator、quantitative_router、topology、verifier、trace、backends。
  - 测试覆盖 CLI、router、graph、trace、memory、orchestrator。
  - 当前本地测试结果为 49 passed。
  - 项目支持 mock、Ollama、vLLM、DeepSeek 等后端入口。
- 视觉想法：模块地图，加测试通过结果条。
- 页面角色：阶段成果。
- 必需源图片：无。

## Slide 12: 后续工作计划
- 关键点：
  - 接入真实 LLM 后端并评估稳定性。
  - 增强协作图的条件分支、工具调用和失败重试。
  - 设计 single-agent、固定多 Agent、quant graph 路径的对比实验。
  - 完成论文实验章节和系统局限性分析。
- 视觉想法：四阶段路线图。
- 页面角色：计划与收束。
- 必需源图片：无。
