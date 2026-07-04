# AEGRL 补充实验执行计划（AI-DLC Workflow Plan）

生成日期：2026-06-13  
工作流：AWS AI-DLC Inception / Workflow Planning  
需求来源：`review_response/aidlc/requirements.md`

## 1. Detailed Analysis Summary

### Transformation Scope

- 类型：Research experiment enhancement，不是系统架构迁移。
- 当前工程：Brownfield Python/SUMO/Stable-Baselines3 实验仓库。
- 主要变化：实验编排、统计分析、结果汇总、论文表述澄清。
- 代码影响：后续阶段可能新增分析脚本和实验脚本，但本计划本身不修改训练/evaluation 核心代码。

### Change Impact Assessment

| Impact Area | Assessment |
|---|---|
| User-facing changes | 无产品用户界面影响；影响论文审稿人阅读和返修说服力。 |
| Structural changes | 低到中。可能新增 `review_response/scripts` 和结果目录，不改主模块结构。 |
| Data model changes | 无数据库变化；需要统一 seed-level CSV 和 summary CSV schema。 |
| API changes | 无公开 API 变化；可能给分析脚本定义 CLI 参数。 |
| NFR impact | 主要是可复现性、统计可靠性、计算成本控制。 |

### Risk Assessment

- 总体风险：Medium。
- 最高风险项：Demo source agnosticism 和 SAC adversary backbone。
- 主要风险来源：训练成本、已有模型路径缺失、SAC adversary 需要新增算法设计、第三场景 victim model 可能不存在。
- 回滚复杂度：Easy。新增文件和结果可独立管理，不影响主代码。
- 测试复杂度：Complex。多 seed、多 env、多 victim、多 source 组合需要分阶段 gate。

## 2. Workflow Decision

### AI-DLC Phases

| Phase / Stage | Decision | Rationale |
|---|---|---|
| Workspace Detection | Completed | 已确认当前为 Brownfield 实验仓库，并已有 `review_response` 返修材料。 |
| Reverse Engineering | Skip | 已在前序工作中检查关键入口：`advTrain.py`、`evaluation_v3.py`、`expert_imitation_learning_MoE.py`、`algorithm.py`、环境注册文件。 |
| Requirements Analysis | Completed | 已生成 `review_response/aidlc/requirements.md`。 |
| User Stories | Skip | 这是论文返修实验规划，不涉及产品用户角色或交互旅程。 |
| Workflow Planning | Completed | 本文件即执行计划。 |
| Application Design | Skip | 不新增应用架构或服务层。 |
| Units Generation | Execute lightweight | 用 ERU-01 到 ERU-09 作为实验工作单元。 |
| Functional Design | Execute per unit if coding starts | 仅当后续实现分析脚本或训练脚本时，为对应 ERU 细化。 |
| NFR Requirements / Design | Execute lightweight | 聚焦可复现性、统计可靠性、计算成本。 |
| Infrastructure Design | Skip | 不涉及部署基础设施。 |
| Code Generation | Later | 本阶段只形成需求和计划；后续按 ERU 实现脚本。 |
| Build and Test | Later | 后续对脚本和实验结果做验证。 |

## 3. Execution Sequence

### Phase 0：准备与基线澄清

#### Step 0.1：统一 baseline 命名

- 工作单元：ERU-05。
- 行动：
  - 所有后续文档、表格和计划统一写 `Vanilla ([14])`。
  - 论文 Section 5.2.2 后续应明确 Vanilla 即 [14] baseline。
  - Table 3 方法名或表注应显式出现 `[14]`。
- 输出：
  - baseline 澄清段落。
  - response letter 对 R3-C10 的回复。
- Gate：
  - 不再出现“补跑独立 [14] baseline”的任务。

#### Step 0.2：建立结果目录约定

- 建议目录：
  - `review_response/results/seed_level/`
  - `review_response/results/summary/`
  - `review_response/figures/`
  - `review_response/commands/`
- 统一字段：
  - `experiment_id`
  - `env`
  - `victim`
  - `source_victim`
  - `method`
  - `seed`
  - `epsilon`
  - `adv_steps`
  - `collision_rate`
  - `mean_attack_times`
  - `episode_count`
  - `model_path`
  - `command`

## 4. P0 Critical Path

### P0-1：统计显著性与例外解释

- 对应需求：FR-002。
- 依赖：已有或补跑主结果 seed-level 数据。
- 执行顺序：
  1. 汇总现有 `evaluation_result` 和 `logs/adv_eval`。
  2. 补齐缺失 seed。
  3. 计算 mean、std、95% CI。
  4. 执行 paired t-test 或 Wilcoxon。
  5. 执行 Holm correction 和 effect size。
  6. 生成 Env-2 例外诊断。
- 输出：
  - `statistical_significance_seed_level.csv`
  - `statistical_significance_summary.csv`
  - manuscript-ready significance table。
- Gate：
  - 每个主结果 cell 至少 5 seeds。
  - PPO Env-2 和 TD3 Env-2 有明确诊断结论。

### P0-2：MoE 专门化与攻击多样性

- 对应需求：FR-005。
- 依赖：MoE 模型、demo/evaluation trajectories。
- 执行顺序：
  1. 读取 `expert_imitation_learning_MoE.py` 的 `Actor.gating_network`。
  2. 对 demo states 和 evaluation states 计算 router weights。
  3. 按 env、victim、phase、attack/idle 分组。
  4. 计算 router entropy、dominant expert ratio、inter-expert output distance。
  5. 输出 heatmap 和多样性指标表。
- 输出：
  - router activation heatmap。
  - entropy/dominant expert table。
  - attack timing diversity plot。
- Gate：
  - 至少一张可放入论文或 appendix 的图。
  - 若 specialization 不明显，正文 claim 必须降调。

### P0-3：Cross-victim prior transfer

- 对应需求：FR-003。
- 依赖：PPO/SAC/TD3 source demonstrations 和 trained victim models。
- 执行顺序：
  1. 为 PPO/SAC/TD3 分别准备 source-specific expert data。
  2. 分别训练 source-specific MoE prior。
  3. 在 Env-1 中训练或评估 AEGRL source-target 组合。
  4. 生成 3x3 transfer matrix。
- 输出：
  - `cross_victim_transfer_env3_3x3.csv`
  - 3x3 CR/AE 表。
- Gate：
  - 9 个 source-target cell 完整。
  - 明确 diagonal 和 off-diagonal 的性能保持比例。

### P0-4：Demo source agnosticism

- 对应需求：FR-004。
- 依赖：至少两类 contrasting demo sources。
- 执行顺序：
  1. 固定 Env-1 PPO victim。
  2. 准备 `Vanilla ([14])` source。
  3. 生成 heuristic random-time/random-direction source。
  4. 生成 weak FGSM 或 weak sequential source。
  5. 各自训练 MoE prior。
  6. 各自训练/评估 AEGRL。
- 输出：
  - source quality table。
  - demo source vs AEGRL CR/AE table。
- Gate：
  - 至少完成 `Vanilla ([14])` 和 heuristic 两类 source。
  - 根据结果决定 agnostic claim 是保留、限制还是降调。

## 5. P1 Supporting Evidence

### P1-1：Hyperparameter sensitivity

- 对应需求：FR-007。
- 执行顺序：
  1. 优先复用 `Scripts/kTest.sh` 分析 `k`。
  2. 比较 Full、Fix、wo_beta。
  3. 复用 `Scripts/MoE_experiment.sh` 分析 sampling ratio。
  4. 如时间允许，分析 ensemble count 和 experts per MoE。
- 输出：
  - `k` sensitivity curve。
  - beta mechanism comparison table。
  - MoE/data ratio sensitivity plot。
- Gate：
  - 至少完成 `k` 和 beta/fix/no-beta 两类结果。

### P1-2：Cost-benefit overhead

- 对应需求：FR-008。
- 执行顺序：
  1. 对 Vanilla ([14])、VPRL、PCRL、AEGRL 记录训练时间。
  2. 统计 expert prior 和 adversary policy 参数量。
  3. 统计 100 episodes evaluation time。
  4. 计算 CR/AE gain per training hour。
- 输出：
  - cost-benefit table。
  - overhead vs CR scatter。
- Gate：
  - 至少报告 AEGRL 与 Vanilla ([14]) 的成本和收益。

## 6. P2 Optional Coverage

### P2-1：SAC adversary backbone sanity check

- 对应需求：FR-009。
- 执行路线：
  - 尝试 Env-1 PPO victim 单配置 SAC-AEGRL。
  - 若算法实现风险过高，执行文字降级方案。
- Gate：
  - SAC-AEGRL 能稳定训练并评估，才进入论文实验。
  - 否则明确写为 limitation/future work。

### P2-2：第三场景扩展

- 对应需求：FR-010。
- 执行路线：
  - 默认 `TrafficEnv7-v0`。
  - 先确认 victim model、state dimension、attack process 兼容。
  - 最小矩阵：AEGRL vs Vanilla ([14])。
- Gate：
  - 至少一个 victim 完整闭环才进入论文。

## 7. Dependency and Parallelization Plan

| Work Unit | Depends On | Can Parallelize With | Notes |
|---|---|---|---|
| ERU-05 Vanilla ([14]) clarification | None | All units | 立即完成，影响命名和论文表述。 |
| ERU-01 Statistics | Existing result CSV | ERU-02 | 最先执行，影响主 claim。 |
| ERU-02 MoE diversity | MoE model and trajectories | ERU-01 | 分析型，成本低。 |
| ERU-03 Cross-victim transfer | Source-specific priors | ERU-04 partially | 训练成本中等。 |
| ERU-04 Demo source | New demo data | ERU-03 partially | 训练成本高。 |
| ERU-06 Hyperparameters | Stable baseline settings | ERU-07 | P1 支撑证据。 |
| ERU-07 Cost-benefit | Training/eval logs | ERU-06 | 可边跑边记录。 |
| ERU-08 SAC backbone | Algorithm extension | None | 高风险，最后做。 |
| ERU-09 New scenario | Victim model compatibility | None | 中高风险，最后做。 |

## 8. Quality Gates

### Gate 0：概念边界检查

- `Vanilla ([14])` 只表示 strict-budget baseline。
- `[14]-derived relaxed-budget demonstrations` 只表示 expert training data source。
- 所有表格和 response letter 必须避免把这两个对象混为一谈。

### Gate A：P0 数据完整性

- 所有 P0 表格有 seed-level CSV。
- 所有主对比使用同一 evaluation episode 数。
- 缺失 cell 明确标注原因，不隐性跳过。

### Gate B：统计可靠性

- 主结果包含 95% CI。
- 关键对比包含 corrected p-value。
- 所有不显著或反向结果进入讨论，不用模糊措辞遮盖。

### Gate C：论文可写入性

- 每个实验产出至少对应一个 manuscript section/table/figure。
- 每个 reviewer issue 有 response sentence。
- Vanilla ([14]) 命名一致。

### Gate D：成本控制

- P0 未完成前，不启动 P2 高风险实现。
- Demo source 和 transfer 可先做 Env-1 PPO 最小闭环，再扩展。
- Cross-victim transfer 先允许 expert-only transfer 或少量 off-diagonal AEGRL 作为 feasibility check，再决定是否跑完整 3x3 x 5 seeds。
- Demo source agnosticism 先比较两类 source，确认 heuristic source 数据质量足够后再加入第三类 source。

### Gate E：证据强度检查

- MoE router heatmap 只能作为 behavioral evidence，不能单独声称完全排除参数量 confound。
- 5 seeds 显著性检验必须与 CI 和 effect size 一起解释。
- SAC adversary backbone 默认不进入 P0；若不做，必须同步降调 manuscript claim。

## 9. Planned Deliverables

| Deliverable | Path / Name | Owner Stage |
|---|---|---|
| AI-DLC requirements | `review_response/aidlc/requirements.md` | Completed |
| AI-DLC execution plan | `review_response/aidlc/execution_plan.md` | Completed |
| Updated experiment docs | `review_response/experiments/*.md` | Existing |
| Seed-level results | `review_response/results/seed_level/*.csv` | Future |
| Summary tables | `review_response/results/summary/*.csv` | Future |
| Figures | `review_response/figures/*` | Future |
| Reproducible commands | `review_response/commands/*.md` | Future |

## 10. Text Workflow Representation

No Mermaid diagram is used, to avoid parser issues.

Text workflow:

1. Clarify baseline naming: Vanilla equals [14].
2. Complete P0 statistics and exception diagnosis.
3. Complete P0 MoE specialization/diversity analysis.
4. Complete P0 cross-victim transfer.
5. Complete P0 demo source agnosticism.
6. Complete P1 hyperparameter sensitivity.
7. Complete P1 cost-benefit overhead.
8. Decide whether P2 SAC adversary and third scenario are feasible.
9. Convert accepted results into manuscript edits and response letter.

## 11. Approval and Next-Step Recommendation

Recommended next implementation step:

1. Start with ERU-05 baseline clarification text because it is zero-compute and fixes a known reviewer misunderstanding.
2. Then implement ERU-01 statistical analysis script and result aggregation.
3. In parallel, implement ERU-02 MoE router analysis because it is low-cost and high payoff.
4. Before running ERU-03 or ERU-04 full experiments, review `review_response/aidlc/risk_register.md` and execute the minimum-loop version first.

## 12. Validation Summary

- No Mermaid or ASCII diagrams included.
- All work units trace to requirements.
- High-risk units have downgrade paths.
- Baseline identity updated to `Vanilla ([14])`.
