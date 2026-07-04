# P0 修改计划：Vanilla 与 [14] Baseline 等价澄清

## 目标审稿意见

- Reviewer 3-C10：审稿人认为最相关的 sequential attack work [14] 只作为 demonstration generator，没有作为直接 baseline。

## 已知事实

作者确认：[14] baseline 实际上就是论文中的 Vanilla。

因此本项不再规划新增 [14] direct baseline 实验。核心任务改为消除命名与表述误解：让审稿人一眼看出 Vanilla 对应 [14] 在相同 bounded sequential attack setting 下的实现。

## 要验证或澄清的 claim

AEGRL 已经与 [14] baseline 进行了同设置比较，只是原稿中将其命名为 Vanilla，导致读者误以为 [14] 没有进入 baseline 表。

## 修改范围

### Section 5.2.2 Baselines

将 Vanilla baseline 的定义改为更明确的表述：

- 原含义：same DRL-based adversarial training framework as AEGRL, but removes all expert-guided components。
- 建议新表述：`Vanilla ([14])` denotes the bounded sequential DRL-based attacker following [14], implemented under the same environment, perturbation magnitude, and attack budget as AEGRL, with all expert-guided components removed.

注意：如果 [14] 的原方法不仅是“去掉 expert guidance”，还包含特定 FGSM/BIM perturbation 生成或 timing policy，需要在文字中逐条对应。

### Table 3 与其他结果表

推荐二选一：

1. 将方法名从 `Vanilla` 改为 `Vanilla ([14])`。
2. 保持 `Vanilla`，但在表注中写明：`Vanilla is the implementation of the bounded sequential attack baseline in [14] without expert guidance.`

推荐使用方案 1，因为审稿人已经明确提出缺少 [14] baseline，方法名直接显示 `[14]` 最稳。

### Related Work

补充一句桥接说明：

- [14] 是本工作的直接 sequential attack baseline。
- AEGRL 沿用其 bounded sequential adversarial evaluation setting，但进一步引入 imperfect expert prior construction、KL guidance 和 adaptive annealing。

### Response Letter

回复重点不是“新增实验”，而是“澄清已有 baseline 身份，并在修订稿中改名/表注”。

## 代码/方法对应表

后续写 response 时建议加入一个小表：

| [14] 机制 | 本文 Vanilla 对应实现 | 说明 |
|---|---|---|
| Bounded sequential attack | Vanilla adversary training/evaluation | 与 AEGRL 使用相同 env、epsilon、attack budget。 |
| No expert guidance | Vanilla removes MoE prior、KL regularization、annealing | 因此是 AEGRL 的 no-expert baseline，也是 [14] baseline。 |
| Same victim policies | PPO/SAC/TD3 victim settings | Table 3 中已覆盖。 |
| Same evaluation protocol | 5 seeds、100 evaluation episodes | 与主实验一致。 |

## 输出表/图

不新增实验表，改为修改现有表格标注：

- Table 3 方法名：`Vanilla ([14])`。
- Baseline paragraph：补充 [14] 对应说明。
- Response letter：明确“we have clarified that Vanilla corresponds to [14]”。

## 验收标准

- 修订稿中不再出现审稿人可能理解为“[14] 只用于 demo generation”的表述。
- Baseline 列表中显式出现 `[14]`。
- Response letter 明确说明这是澄清而非新增实验，并指出修订位置。
- 成本收益、新场景扩展、统计分析等后续文档均使用 `Vanilla ([14])` 作为 baseline 名称。

## 论文对应修改位置

- Section 5.2.2 Baselines。
- Table 3 方法名称或表注。
- Section 2.2 Related Work。
- Response letter 对 Reviewer 3-C10 的回复。

## Rebuttal 回应句

感谢审稿人指出 baseline 表述不清的问题。我们澄清并在修订稿中显式标注：原稿中的 Vanilla baseline 即 [14] 的 bounded sequential attack baseline 在相同设置下的实现，其去除了 AEGRL 的 expert-guided components；因此 Table 3 已包含与 [14] 的直接比较。

