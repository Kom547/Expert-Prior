# P2 实验计划：新增交通场景扩展

## 目标审稿意见

- Reviewer 3-C11：实验只覆盖两个 traffic scenarios，general effectiveness 的证据较窄。

## 要验证的 claim

AEGRL 在除 unprotected left-turn 和 on-ramp merging 外的第三类交通场景中仍具有一定 failure discovery 能力。

## 场景选择

优先级：

1. `TrafficEnv7-v0`：工程中已有 `Environment/environment/env7/highway.*`，适合作为 highway/lane-related 场景。
2. `TrafficEnv5-v0`：Navigation 场景，但 state/action 维度可能不同，代码适配风险更高。
3. `TrafficEnv6-v0`：Intersection_3 变体，可作为补充但与 Env-1 差异可能不够大。

默认选择：`TrafficEnv7-v0`。

## 实验设置

- 环境：`TrafficEnv7-v0`。
- Victim：优先 PPO；若已有 SAC/TD3 victim model 可直接加载，再扩展。
- Methods：
  - Vanilla ([14])。
  - AEGRL。
- Epsilon：0.05。
- Budget：Gamma=4。
- Seeds：0..4。
- Evaluation episodes：100。

## 代码入口

- 环境注册：`Environment/environment/__init__.py`
- Victim training：`train.py` 或现有 victim model 路径。
- Demo collection：`evaluation_v3.py --expert_recording`
- Expert training：`expert_imitation_learning_MoE.py`
- AEGRL training：`advTrain.py`
- Evaluation：`evaluation_v3.py`

## 关键检查

- `TrafficEnv7-v0` observation dimension 是否仍兼容 `state_shape=(28,)` expert prior 输入。
- victim model 是否存在于 `logs/eval/TrafficEnv7-v0/<algo>/`。
- attack env 是否正确维护 `attack_remain / adv_steps`。
- FGSM/PGD perturbation 对该环境状态维度是否适配。

## 输出表/图

- 单场景对比表：Vanilla ([14])、AEGRL 的 CR/AE。
- 如果只完成 PPO victim，则明确标注为 additional scenario sanity check。
- 可选图：第三场景中攻击时刻分布。

## 统计方法

- 5 seeds mean ± std 和 95% CI。
- AEGRL vs Vanilla ([14]) 做 paired test。

## 验收标准

- 至少完成一个新增场景和一个 victim policy 的完整闭环。
- 若结果不理想，正文不能将其包装成强泛化证据，应放入 limitation 并说明复杂场景扩展仍需更多研究。

## 论文对应修改位置

- Section 5.1：新增场景描述。
- Section 5.5 或 Appendix：新增 scenario expansion 表。
- Conclusion：将 generality claim 改为“validated across three representative scenarios”或更保守表述。

## Rebuttal 回应句

我们新增了一个 highway 场景的补充实验，用最小矩阵评估 AEGRL 和 Vanilla ([14])，以扩展原稿仅含两个场景的实验覆盖范围。
