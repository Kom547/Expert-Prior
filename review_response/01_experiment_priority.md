# AEGRL 补充实验优先级评估

生成日期：2026-06-11  
策略：全覆盖强回应，但按优先级依次执行，避免先投入到高成本低命中的实验。

## 评分规则

每项实验按 6 个维度评分，1 分最低，5 分最高。

| 维度 | 含义 |
|---|---|
| 审稿人数量 | 有多少位审稿人直接或间接提出该问题。 |
| 严重性 | 是否影响 Major Revision 的核心判断。 |
| 核心 claim 支撑 | 是否直接支撑 AEGRL 的主要贡献。 |
| 实现成本 | 5 表示低成本，1 表示高成本。 |
| 表图产出 | 是否容易形成清晰的新表格或图。 |
| 返修信服力 | 是否能显著增强 rebuttal 和 revised manuscript 的说服力。 |

总分越高，优先级越高。实现成本采用反向分：越容易实现分越高。

## 优先级总表

| 优先级 | 实验 | 对应文档 | 审稿人数量 | 严重性 | 核心 claim | 实现成本 | 表图产出 | 返修信服力 | 总分 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 统计显著性与例外解释 | `experiments/P0_statistical_significance_and_exceptions.md` | 2 | 5 | 4 | 4 | 5 | 5 | 25 |
| P0 | Cross-victim prior transfer | `experiments/P0_cross_victim_prior_transfer.md` | 1 | 5 | 5 | 3 | 5 | 5 | 24 |
| P0 | Demo source agnosticism | `experiments/P0_demo_source_agnosticism.md` | 2 | 5 | 5 | 2 | 5 | 5 | 24 |
| P0 | MoE 专门化与多样性 | `experiments/P0_moe_specialization_diversity.md` | 2 | 4 | 5 | 4 | 5 | 5 | 25 |
| P0 | Vanilla/[14] 等价澄清 | `experiments/P0_vanilla_14_clarification.md` | 1 | 5 | 4 | 5 | 4 | 5 | 24 |
| P1 | 超参数敏感性 | `experiments/P1_hyperparameter_sensitivity.md` | 1 | 4 | 4 | 3 | 5 | 4 | 21 |
| P1 | 成本收益与复杂度 | `experiments/P1_cost_benefit_overhead.md` | 1 | 4 | 3 | 4 | 4 | 4 | 20 |
| P2 | SAC adversary backbone sanity check | `experiments/P2_adversary_backbone_sac.md` | 1 | 4 | 3 | 1 | 3 | 3 | 15 |
| P2 | 新场景扩展 | `experiments/P2_scenario_expansion.md` | 1 | 3 | 3 | 2 | 4 | 3 | 16 |

## 执行顺序

1. 先做统计显著性与例外解释。它直接修复 Reviewer 2 和 Reviewer 3 对结果可靠性的质疑，也能帮助决定正文 claim 如何降调。
2. 并行推进 MoE 专门化与多样性分析。该实验主要是分析型，复用已训练 MoE，成本低，图表回报高。
3. 做 cross-victim prior transfer。它是 Reviewer 1 认为最严重的 generality 缺口之一，3x3 表非常适合 rebuttal。
4. 做 demo source agnosticism。它直接回应两位审稿人，但需要生成新 demo、训练新 prior 和 AEGRL，计算成本较高。
5. 做 Vanilla/[14] 等价澄清。不再额外跑 [14] direct baseline，而是把 Vanilla 明确重命名或表注为 [14] baseline，并补充方法对应说明。
6. 做超参敏感性与成本收益。两者用于支撑 practical significance 和 design choice。
7. 最后做 SAC adversary backbone 与新增场景。它们覆盖剩余意见，但实现风险和收益比低于 P0/P1。

## P0 完成标准

P0 阶段完成后，返修稿至少应能新增：

- 一张统计显著性表或在主结果表中加入 CI/p-value 标注。
- 一张 cross-victim transfer 3x3 表。
- 一组 MoE router 专门化热力图或 entropy/dominant ratio 表。
- 一张 demo source 对比表。
- 清晰说明 Vanilla/[14] baseline 的对应关系，并在主结果表或表注中消除审稿人误解。

## 风险与降级策略

| 风险 | 降级策略 |
|---|---|
| demo source 训练时间过长 | 先只在 Env-1、PPO victim、5 seeds 上跑最小闭环；若趋势清楚，再扩展 SAC/TD3。 |
| cross-victim 需要大量 retraining | 优先复用 source-specific expert prior，AEGRL training 保持原默认设置；必要时先做 expert-only transfer，再做 AEGRL transfer。 |
| 审稿人继续认为缺少 [14] | 在 response letter 中明确说明 Vanilla 就是 [14] baseline，并在修订稿 Table 3 和 Section 5.2.2 同步改名为 `Vanilla ([14])` 或 `Vanilla / [14] baseline`。 |
| SAC adversary 实现成本过高 | 明确将 manuscript claim 改为 PPO instantiation，并把 SAC/TD3 adversary 放到 future work。 |
