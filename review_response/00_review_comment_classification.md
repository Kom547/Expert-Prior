# AEGRL 审稿意见分类文档

生成日期：2026-06-11  
材料来源：三位审稿人的 Major Revision 意见、原始手稿 `EGRL_0321.pdf`、当前工程代码结构。

## 分类原则

本文件按意见的相似性归类，并标注主要处理方式：

- 文字修改：主要通过论文表述、定义、公式说明、相关工作或 response letter 解释解决。
- 实验修改：需要新增实验、重新统计、补充图表或生成新结果。
- 混合修改：既需要实验结果，也需要论文叙述调整。

## 总体判断

三位审稿人总体认可问题动机、AEGRL 框架和现有实验完整性，但 Major Revision 的核心压力集中在四类证据缺口：

1. 专家先验是否真的具有泛化性：demo source 泛化、cross-victim transfer。
2. MoE 是否真的捕获了多样化攻击模式：router 专门化和攻击多样性缺少直接证据。
3. 结果是否统计可靠：缺少显著性检验，且部分 Env-2 结果与“almost all cases”表述冲突。
4. 方法复杂度是否值得：需要澄清 Vanilla 即 [14] baseline，并补充超参敏感性、成本收益分析。

## A. 专家来源与先验泛化

| 子问题 | 关联意见 | 类型 | 当前问题 | 建议处理 |
|---|---|---:|---|---|
| Demo source agnosticism | R1-C1, R3-C6 | 实验修改 | 手稿声称 expert generation module 不依赖 demonstration source，但所有实验都使用 [14] relaxed-budget attacker 生成的数据。 | 新增至少三类 demo source：[14]、heuristic random-time/random-direction、weak sequential 或 myopic FGSM；分别训练 MoE prior，并评估 AEGRL 的 CR/AE。 |
| Cross-victim prior transfer | R1-C4 | 实验修改 | Table 3 中每个 victim 的 expert prior 都来自同 victim 或混合 victim 数据，无法说明 PPO/SAC/TD3 之间的 prior transfer。 | 在 Env-1 构造 3x3 transfer table：source victim 为 PPO/SAC/TD3，target victim 为 PPO/SAC/TD3。 |
| Imperfect expert 定义 | R3-C1 | 文字修改 | “imperfect expert”是核心动机，但没有明确定义 imperfection 类型。 | 在 Introduction 或 Method 前新增定义：coverage insufficiency、behavioral bias、budget mismatch、victim-coupling mismatch。 |
| Distribution shift | R3-C5 | 文字修改 | BC expert 在 demonstration distribution 上训练，却在 adversary rollout distribution 下提供 KL guidance。 | 在 Method 或 Discussion 中说明 KL guidance 的局限、annealing 对 distribution shift 的缓解作用及剩余风险。 |

## B. MoE 专门化与攻击多样性

| 子问题 | 关联意见 | 类型 | 当前问题 | 建议处理 |
|---|---|---:|---|---|
| MoE experts 是否专门化 | R1-C3 | 实验修改 | 现有 S-MLP < S-MoE < Ens-MoE 的 CR 排名不能排除参数量、ensemble smoothing 或 regularization 的影响。 | 读取 `gating_network` 权重，输出 router activation heatmap、dominant expert ratio、router entropy。 |
| 攻击模式多样性 | R3-C8 | 实验修改 | CR 只能说明攻击成功率，不能说明发现了更多 failure modes。 | 使用 attack timing、attack state cluster、router activation cluster 或 state-space coverage 指标补充多样性证据。 |
| MoE 架构选择依据 | R3-C4 | 混合修改 | 5 个 MoE、每个 3 experts 缺少敏感性分析。 | 将 MoE 数量、experts 数量纳入超参敏感性计划；正文中补充经验选择依据。 |

## C. 结果可靠性与例外讨论

| 子问题 | 关联意见 | 类型 | 当前问题 | 建议处理 |
|---|---|---:|---|---|
| 统计显著性 | R2 Negative, R3-C7 | 混合修改 | 现有表格只报告 mean ± std，缺少 p-value、CI、效应量。 | 对 5 seeds 聚合结果做 95% CI、paired t-test 或 Wilcoxon、Holm correction、effect size。 |
| Env-2 例外结果 | R2 Negative | 混合修改 | TD3 Env-2 中 AEGRL CR 低于 Vanilla/VPRL；PPO Env-2 与 Vanilla 打平且方差更大。 | 重新检查 seed-level 结果，并在正文中承认例外，讨论 expert mismatch、scenario dynamics、annealing schedule。 |
| “almost all cases”措辞 | R2 Negative | 文字修改 | 当前表述过强，与 Table 3 例外不一致。 | 改为“in most settings”或按 metric/scenario 精确描述。 |

## D. Baseline、复杂度与实用性

| 子问题 | 关联意见 | 类型 | 当前问题 | 建议处理 |
|---|---|---:|---|---|
| Vanilla 与 [14] 等价澄清 | R3-C10 | 文字修改为主 | 审稿人误以为 [14] 只作为 demonstration generator，实际上论文中的 Vanilla 就是 [14] baseline。当前问题是 baseline 命名和说明不够清楚。 | 不再规划额外跑 [14] direct baseline；改为在 Section 5.2.2、Table 3 表注和 response letter 中明确说明 Vanilla corresponds to [14] under the same bounded sequential attack setting，并给出代码/方法对应依据。 |
| Cost-benefit | R3-C9 | 混合修改 | 改进幅度较 modest，但没有训练时间、推理延迟、参数量开销。 | 补充成本收益表：CR/AE gain、training time、inference latency、parameter count、expert model count。 |
| 核心贡献是组合而非创新 | R3-C2 | 混合修改 | MoE、KL regularization、annealing 均为已有技术，缺少组合带来的 emergent property 证据。 | 用 ablation、transfer、diversity、negative transfer mitigation 组织为“组合解决 imperfect prior 两阶段问题”的证据链。 |

## E. Adversary Backbone 与场景覆盖

| 子问题 | 关联意见 | 类型 | 当前问题 | 建议处理 |
|---|---|---:|---|---|
| “not restricted to PPO” | R1-C2 | 实验或文字降调 | 当前 `advTrain.py` 的 adversary training 路径是 PPO 系；Table 3 的 PPO/SAC/TD3 是 victim，不是 adversary backbone。 | 全覆盖策略下尝试 Env-1 单配置 SAC-backbone adversary sanity check；若成本过高，正文明确改为“we instantiate AEGRL with PPO”。 |
| 新场景扩展 | R3-C11 | 实验修改 | 手稿只有两个 traffic scenarios。 | 追加一个低优先级场景，优先 `TrafficEnv7-v0` highway，跑 AEGRL vs Vanilla 最小矩阵；表注说明 Vanilla 即 [14] baseline。 |

## F. 数学、威胁模型与可复现性文字修改

| 子问题 | 关联意见 | 类型 | 当前问题 | 建议处理 |
|---|---|---:|---|---|
| Threat model | R2 Negative | 文字修改 | “relevant information”和 BIM access 表述不够精确。 | 明确 attacker 知识：victim observation、normalization、reward/dynamics access、perturbable dimensions、temporal consistency。 |
| 数学符号统一 | R2 Negative | 文字修改 | Eq. (1) 中 `a^adv`、`u^adv`、`a'_t` 混用；Eq. (3) 符号正负需解释。 | 统一 adversarial target action 和 victim action 符号，明确 safety penalty 与 adversarial reward 的关系。 |
| Theorem 1 | R2 Negative, R3-C3 | 文字修改 | 证明假设不完整，且可能被认为 circular。 | 改为 Proposition 或 Motivation Lemma；补充 differentiability、support、policy class 假设，弱化理论贡献表述。 |
| Eq. (26) 负 gap | R1-C5 | 文字修改 | 当前 policy outperform reference 时公式未定义。 | 明确使用 clipped gap：`max(0, reference - current)` 或说明 beta 下界与处理逻辑。 |
| `n_t` 标注规则 | R1-C6 | 文字修改 | demo relaxed budget 与 deployment budget 的 remaining budget 标签规则未说明。 | 补充 `Gamma'` 值，并说明代码中状态使用 `attack_remain / adv_steps`，demo 转换时按各自 trajectory 的剩余预算归一化。 |
| 相关工作 | R3-C12 | 文字修改 | suboptimal/imperfect demonstrations 文献不足。 | 增加 learning from suboptimal demonstrations、imperfect demonstrations、KL-regularized RL 相关文献。 |
| 代码和数据可用性 | R3 Weakness 9 | 文字修改 | 无 public code/data commitment。 | 增加 availability statement：接收后公开核心代码、配置、seed-level results；受限制数据给生成脚本。 |

## 审稿意见覆盖检查

所有明确编号意见均已覆盖：

- Reviewer 1：C1-C6 均覆盖。
- Reviewer 2：threat model、notation、Theorem、statistics/exceptions 均覆盖。
- Reviewer 3：C1-C12 均覆盖。
