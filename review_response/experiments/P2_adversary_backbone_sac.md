# P2 实验计划：SAC Adversary Backbone Sanity Check

## 目标审稿意见

- Reviewer 1-C2：手稿声称 AEGRL 不限于 PPO，但 Eq. (17) 是 clipped-PPO objective，当前 adversary training 也只实现 PPO 系路径。

## 要验证的 claim

AEGRL 的 expert-guided idea 是否可以迁移到 SAC-style stochastic actor critic adversary；如果不能在返修周期内可靠完成，则论文应明确收缩 claim。

## 当前代码事实

- `advTrain.py` 的 `create_model` 在 expert attack 下选择 `AdversarialEGPPO`。
- `algorithm.py` 中 `AdversarialEGPPO` 继承 PPO，并在 PPO loss 中加入 expert KL。
- `evaluation_v3.py` 可以加载 SAC model，但主要用于 victim 或已训练 adversary evaluation，不等于 SAC adversary training 已完整实现。

## 实验设置

- 最小 sanity check：
  - 环境：`TrafficEnv3-v5`。
  - Victim：PPO。
  - Adversary backbone：SAC。
  - Expert prior：默认 MoE prior。
  - Epsilon：0.05。
  - Budget：Gamma=4。
  - Seeds：0, 1, 2；若稳定再补齐 5 seeds。
  - Baselines：PPO-AEGRL、SAC-Vanilla、SAC-AEGRL。

## 实现方向

1. 新增 SAC adversary wrapper，复用 adversarial environment interaction 和 `attack_process`。
2. 对 SAC stochastic actor 的 action distribution 定义 KL-to-prior。
3. 将 expert KL 加入 actor loss，而不是 PPO clipped surrogate。
4. 注意 SAC entropy temperature `alpha` 与 expert coefficient `beta` 的相互作用。
5. 保持 evaluation API 与 `evaluation_v3.py` 一致。

## 代码入口

- 参考：`algorithm.py` 中 `AdversarialEGPPO.train`
- 训练入口：后续可扩展 `advTrain.py --adv_algo SAC`
- Evaluation：`evaluation_v3.py --adv_algo SAC`

## 输出表/图

- 单配置 sanity table：PPO-AEGRL vs SAC-Vanilla vs SAC-AEGRL 的 CR/AE。
- 训练稳定性曲线：rollout reward / CR proxy。

## 验收标准

- SAC-AEGRL 能完整训练并评估，不出现明显崩溃或 loss NaN。
- 如果 SAC-AEGRL 优于 SAC-Vanilla，可作为 limited evidence。
- 如果实现不稳定或成本过高，必须执行降级方案。

## 降级方案

若 SAC adversary 不能可靠完成，论文改法如下：

- 删除或弱化 “interchangeable across DRL backbones”。
- 改为：“In this work, we instantiate AEGRL with PPO because the clipped on-policy objective provides a stable interface for KL-based expert regularization. Extending the same idea to entropy-regularized off-policy methods such as SAC requires additional design and is left as future work.”

## 论文对应修改位置

- Section 4.3：说明本工作实例化为 PPO-based adversary。
- Section 5：避免把 victim PPO/SAC/TD3 描述成 adversary backbone 泛化。
- Limitation：讨论 SAC/TD3 adversary extension。

## Rebuttal 回应句

我们澄清了 PPO/SAC/TD3 在原稿中指 victim policies，而非 adversary backbone；同时补充一个 SAC adversary sanity check，或在无法稳定完成时明确将 backbone extension 收缩为 future work。

