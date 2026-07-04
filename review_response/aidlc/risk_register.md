# AEGRL 补充实验风险与问题登记表

生成日期：2026-06-13  
目的：指出当前补充实验需求和执行计划中仍存在的关键风险，避免后续实验跑偏。

## 1. 关键问题总览

| ID | 问题 | 严重性 | 影响 | 处理建议 |
|---|---|---:|---|---|
| RISK-001 | `Vanilla ([14])` baseline 与 `[14] relaxed-budget demo source` 容易混淆。 | High | 会让 response letter 和 demo source 实验逻辑变乱。 | 明确区分：[14] strict-budget baseline = Vanilla；[14] relaxed-budget attacker = demonstration generator。 |
| RISK-002 | P0 demo source agnosticism 计算成本很高。 | High | 三类 source x MoE training x AEGRL training x 5 seeds 可能拖慢返修。 | 先做 Env-1/PPO/2 sources 最小闭环，再决定是否扩展。 |
| RISK-003 | Cross-victim transfer 3x3 AEGRL full training 成本较高。 | High | 9 cells x 5 seeds 若都重训 AEGRL，成本可观。 | 先做 expert-only transfer + 少量 AEGRL off-diagonal，再补完整矩阵。 |
| RISK-004 | 5 seeds 的显著性检验统计功效有限。 | Medium | p-value 不显著不一定说明方法无效。 | 同时报 effect size、CI、seed-level distribution，并谨慎解释。 |
| RISK-005 | MoE router heatmap 只能证明 routing pattern，不足以完全排除容量 confound。 | Medium | Reviewer 1 提到 MoE 参数量/ensemble regularization 混杂。 | 增加 parameter-matched MLP 或在正文承认该分析是 direct behavioral evidence，不是完全因果隔离。 |
| RISK-006 | Heuristic random source 可能产生太少 successful demos。 | High | BC expert 学不到有效 prior，实验结论可能变成 source quality failure。 | 同时报告 source quality；允许使用 mixed successful/near-miss trajectories 或降低成功过滤阈值。 |
| RISK-007 | SAC adversary backbone 是高风险代码实现，不适合 P0。 | High | 容易引入算法不稳定和大量调参。 | 默认作为 P2；若时间紧，直接执行 claim 降调方案。 |
| RISK-008 | 第三场景 `TrafficEnv7-v0` 可能缺 victim model 或 state/action shape 不兼容。 | Medium | 新场景扩展可能卡在工程适配而非论文问题。 | 先做 feasibility check，不通过则不承诺进主文。 |
| RISK-009 | 当前 AI-DLC 文档没有完整生成 `aidlc-state.md` 和 `audit.md`。 | Low | 不完全符合 upstream AI-DLC 项目级流程。 | 当前作为论文返修规划 wrapper 可接受；若要严格 AI-DLC，可补状态和审计文件。 |

## 2. 需要立即修正的概念边界

### 2.1 Vanilla 与 [14] 的关系

必须用以下表述保持一致：

- `Vanilla ([14])`：严格预算设置下的 bounded sequential attack baseline，也就是 Table 3 中的 Vanilla。
- `[14]-derived relaxed-budget demonstrations`：用于训练 expert prior 的 demonstration source，预算更宽松，不等同于 Table 3 的 strict-budget Vanilla evaluation。

这两个对象来自同一类方法思想，但在实验角色上不同：

| 对象 | 实验角色 | 预算设置 | 出现位置 |
|---|---|---|---|
| Vanilla ([14]) | baseline attack method | strict budget, e.g. Gamma=4 | 主结果表、成本收益、新场景对比 |
| [14]-derived demos | expert training data source | relaxed budget | demo source agnosticism、expert prior construction |

## 3. P0 实验降本版建议

如果时间或算力紧张，P0 应按以下最小闭环执行：

1. ERU-05 Vanilla ([14]) 澄清：零计算，立即完成。
2. ERU-01 统计显著性：先聚合已有结果，不补跑。
3. ERU-02 MoE router 分析：先用已有 expert model 和已有 trajectories。
4. ERU-03 Cross-victim transfer：先做 Env-1 PPO/SAC/TD3 expert-only transfer，再挑 2 个 off-diagonal 做 AEGRL training。
5. ERU-04 Demo source：先比较 `[14]-derived demos` 与 heuristic source，只做 Env-1/PPO。

## 4. 论文写作风险

| Claim | 风险 | 建议措辞 |
|---|---|---|
| Expert module is agnostic to demo source | 过强 | “can ingest demonstrations from different sources, while performance depends on source quality and coverage” |
| MoE captures diverse attack patterns | 证据需补强 | “router analysis suggests differentiated expert usage across traffic states” |
| AEGRL best in almost all cases | 已被反例挑战 | “AEGRL improves average performance in most settings, with exceptions discussed below” |
| Not restricted to PPO | 当前无 adversary backbone 证据 | “we instantiate AEGRL with PPO; extension to SAC/TD3 adversaries is future work” |

## 5. 结论

当前计划是可执行的，但不能认为“没有问题”。最需要警惕的是概念混淆和算力爆炸。后续执行应先走最小闭环，用结果决定是否扩展，而不是一次性铺开所有 full matrix。

