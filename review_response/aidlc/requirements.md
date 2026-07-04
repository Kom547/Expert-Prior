# AEGRL 补充实验需求文档（AI-DLC Requirements）

生成日期：2026-06-13  
工作流：AWS AI-DLC Inception / Requirements Analysis  
适用范围：TDSC AEGRL Major Revision 补充实验、统计分析和返修表述澄清。

## 1. Intent Analysis

### 用户请求

使用 `aidlc-workflows` 重新梳理 AEGRL 论文返修阶段需要补充的实验，形成需求文档和 plan。

### 请求类型

- 类型：Enhancement / Research revision planning。
- 项目状态：Brownfield。当前工程已有 SUMO 环境、AEGRL/PPO adversary 训练代码、MoE expert 训练代码、evaluation 脚本，以及前置返修分析文档。
- 范围估计：多实验、多文档、多阶段执行，不要求本阶段修改代码或运行实验。
- 复杂度估计：Complex。原因是审稿意见涉及 claim 证据链、统计可靠性、泛化性、实验成本和论文表述一致性。

## 2. Business Context

### 返修目标

在 Major Revision 中用最少的无效计算成本，最大程度回应审稿人对 AEGRL 的关键质疑，形成可执行、可验证、可写入论文和 response letter 的补充实验计划。

### 成功标准

- 每条高影响审稿意见都有明确处理路径：实验、文字澄清或二者结合。
- P0 实验优先覆盖影响论文接受风险最大的证据缺口。
- 所有新增实验都有明确输入、代码入口、输出表图、统计方法和验收标准。
- 明确修正一个关键事实：论文中的 `Vanilla` 就是 [14] baseline，不再重复规划独立 [14] baseline 实验。

## 3. Functional Requirements

| ID | Requirement | Priority | Source | Acceptance Criteria |
|---|---|---:|---|---|
| FR-001 | 系统性重组所有补充实验需求，并按 P0/P1/P2 排序。 | Must | R1/R2/R3 综合 | 文档中每个实验都有优先级、审稿意见来源、目标 claim 和完成标准。 |
| FR-002 | 将统计显著性与例外结果解释列为最高优先级之一。 | Must | R2 Negative, R3-C7 | 需求覆盖 CR/AE 的 95% CI、paired t-test 或 Wilcoxon、Holm correction、effect size，并特别处理 PPO Env-2 与 TD3 Env-2。 |
| FR-003 | 规划 cross-victim prior transfer 实验。 | Must | R1-C4 | 明确 Env-1 3x3 source-target matrix：source victim 和 target victim 均为 PPO/SAC/TD3。 |
| FR-004 | 规划 demo source agnosticism 实验。 | Must | R1-C1, R3-C6 | 至少包含当前 [14]/Vanilla source、heuristic random-time random-direction source、weak FGSM 或 weak sequential source。 |
| FR-005 | 规划 MoE specialization 与 attack diversity 分析。 | Must | R1-C3, R3-C8 | 明确读取 `gating_network`，输出 router activation、router entropy、dominant expert ratio 和 attack timing/state diversity。 |
| FR-006 | 明确 Vanilla 与 [14] baseline 的等价澄清，不再规划额外 [14] direct baseline。 | Must | 用户新确认, R3-C10 | 文档中统一使用 `Vanilla ([14])`，并要求在 baseline 定义、Table 3 表注和 response letter 中澄清。 |
| FR-007 | 规划 hyperparameter sensitivity。 | Should | R3-C4 | 覆盖 annealing `k`、fixed/no-beta、MoE ensemble count、experts per MoE、sampling ratio。 |
| FR-008 | 规划 cost-benefit overhead 分析。 | Should | R3-C9 | 覆盖训练时间、评估时间、推理延迟、参数量、CR/AE gain 和 failure discovery efficiency。 |
| FR-009 | 规划 SAC adversary backbone sanity check，并给出降级策略。 | Could | R1-C2 | 明确最小 Env-1 PPO victim 配置；若实现成本过高，要求论文降调为 PPO instantiation。 |
| FR-010 | 规划第三交通场景扩展。 | Could | R3-C11 | 默认选择 `TrafficEnv7-v0` highway，最小矩阵为 AEGRL vs Vanilla ([14])。 |
| FR-011 | 为每项实验定义可复现实验产物。 | Must | Reproducibility | 每项实验需输出 seed-level CSV、summary CSV/table、figure 或 manuscript-ready table，以及命令记录。 |
| FR-012 | 将实验结果与论文修改位置绑定。 | Must | Revision workflow | 每项实验必须标注预计进入的 manuscript section/table/figure 和 response letter 回应句。 |

## 4. Non-Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|---|---|---:|---|
| NFR-001 | 可追踪性 | Must | 每个实验需求能追溯到 reviewer comment 或用户确认事实。 |
| NFR-002 | 可复现性 | Must | 明确 env、victim、seed、epsilon、budget、evaluation episodes、代码入口和输出路径。 |
| NFR-003 | 计算成本控制 | Must | 先执行 P0；P1/P2 只有在 P0 结果稳定后继续。 |
| NFR-004 | 论文表述一致性 | Must | 所有文档统一使用 `Vanilla ([14])`，避免再制造“缺少 [14] baseline”的误解。 |
| NFR-005 | 统计可靠性 | Must | 所有主表新增结果至少报告 mean/std/CI；关键对比报告 corrected p-value 和 effect size。 |
| NFR-006 | 风险降级 | Should | 对高成本或高风险实验提供降级路线，尤其是 SAC adversary backbone 和新增场景。 |

## 5. Experiment Requirement Units

### ERU-01：统计显著性与例外解释

- 优先级：P0。
- 关联需求：FR-002、FR-011、NFR-005。
- 核心问题：现有 Table 3 缺少统计检验，且 Env-2 出现 AEGRL 不占优或方差较大的例外。
- 默认输入：已有 5 seeds evaluation results；缺失时用 `evaluation_v3.py` 补跑。
- 输出：seed-level CSV、significance summary、例外结果诊断表。

### ERU-02：MoE 专门化与攻击多样性

- 优先级：P0。
- 关联需求：FR-005、FR-011。
- 核心问题：CR 不能证明 MoE experts specialize，也不能证明攻击模式多样。
- 默认输入：MoE expert models、demo states、evaluation trajectories。
- 输出：router heatmap、entropy table、dominant expert ratio、attack timing/state diversity plot。

### ERU-03：Cross-victim prior transfer

- 优先级：P0。
- 关联需求：FR-003、FR-011。
- 核心问题：当前 prior 可能只对同 victim demonstrations 有效。
- 默认输入：PPO/SAC/TD3 source-victim demonstrations。
- 输出：Env-1 3x3 transfer matrix。

### ERU-04：Demo source agnosticism

- 优先级：P0。
- 关联需求：FR-004、FR-011。
- 核心问题：当前 agnostic claim 只用单一 source 支撑。
- 默认输入：[14]/Vanilla source、heuristic source、weak source。
- 输出：source quality table、expert-only and AEGRL CR/AE table。

### ERU-05：Vanilla ([14]) baseline 澄清

- 优先级：P0。
- 关联需求：FR-006、NFR-004。
- 核心问题：审稿人误以为 [14] 没有作为 baseline。
- 默认输入：作者确认 Vanilla 即 [14] baseline。
- 输出：baseline 命名修改建议、Table 3 表注、response letter 回应段。

### ERU-06：Hyperparameter sensitivity

- 优先级：P1。
- 关联需求：FR-007。
- 核心问题：MoE 架构和 annealing 参数缺少 justification。
- 输出：`k` sensitivity、beta/fix/no-beta 对比、MoE count 或 sampling ratio sensitivity。

### ERU-07：Cost-benefit overhead

- 优先级：P1。
- 关联需求：FR-008、NFR-003。
- 核心问题：增益 modest，缺少复杂度和开销解释。
- 输出：训练时间、推理延迟、参数量、CR/AE gain 的成本收益表。

### ERU-08：SAC adversary backbone sanity check

- 优先级：P2。
- 关联需求：FR-009、NFR-006。
- 核心问题：原稿不能声称 adversary backbone 不限于 PPO。
- 输出：最小 SAC-AEGRL sanity table，或明确 claim 降调方案。

### ERU-09：第三交通场景扩展

- 优先级：P2。
- 关联需求：FR-010、NFR-006。
- 核心问题：只有两个 traffic scenarios。
- 输出：`TrafficEnv7-v0` 下 AEGRL vs Vanilla ([14]) 的最小对比表。

## 6. Traceability Matrix

| Reviewer Issue | Requirement IDs | Experiment Unit | Handling |
|---|---|---|---|
| R1-C1 demo source agnostic | FR-004 | ERU-04 | 实验为主，文字降调为辅。 |
| R1-C2 not restricted to PPO | FR-009 | ERU-08 | P2 sanity check 或 claim 降调。 |
| R1-C3 MoE specialization | FR-005 | ERU-02 | Router/diversity 分析。 |
| R1-C4 cross-victim transfer | FR-003 | ERU-03 | 3x3 transfer table。 |
| R1-C5 Eq. 26 negative gap | FR-012 | 文本修改 | 不属于补充实验，进入 response/正文修改。 |
| R1-C6 `n_t` construction | FR-012 | 文本修改 | 不属于补充实验，进入方法说明。 |
| R2 statistics/exceptions | FR-002 | ERU-01 | 统计检验和例外讨论。 |
| R2 threat model/notation/theorem | FR-012 | 文本修改 | 不属于补充实验，进入正文修改。 |
| R3-C4 hyperparameters | FR-007 | ERU-06 | P1 sensitivity。 |
| R3-C6 demo source | FR-004 | ERU-04 | P0 source comparison。 |
| R3-C8 diversity | FR-005 | ERU-02 | P0 diversity analysis。 |
| R3-C9 cost-benefit | FR-008 | ERU-07 | P1 overhead analysis。 |
| R3-C10 missing [14] baseline | FR-006 | ERU-05 | 澄清 Vanilla ([14])，不新增独立实验。 |
| R3-C11 more scenarios | FR-010 | ERU-09 | P2 scenario expansion。 |

## 7. Out of Scope

- 本文档不要求立即修改实验代码。
- 本文档不要求立即运行补充实验。
- 本文档不直接生成 response letter 或修改论文正文。
- 本文档不规划独立 [14] baseline 实验，因为用户已确认 Vanilla 即 [14]。

## 7.1 Known Issues and Risk Controls

详细风险见 `review_response/aidlc/risk_register.md`。执行补充实验前必须特别检查：

- `Vanilla ([14])` baseline 与 `[14]-derived relaxed-budget demonstrations` 的实验角色不同，不能混写。
- Demo source agnosticism 和 cross-victim transfer 不应一次性 full matrix 铺开，应先做最小闭环。
- MoE router 分析不能单独完全排除参数量 confound，必要时补 parameter-matched MLP 或在论文中限制 claim。
- 5 seeds 显著性检验统计功效有限，必须结合 CI 和 effect size 解释。
- SAC adversary backbone 默认是高风险 P2，不应影响 P0 返修主线。

## 8. Validation Summary

- Markdown 使用表格和列表，无 Mermaid 或 ASCII 图。
- 所有新增实验需求都有 reviewer traceability。
- 高成本实验均提供降级路径。
- `Vanilla ([14])` 命名已统一纳入需求。
