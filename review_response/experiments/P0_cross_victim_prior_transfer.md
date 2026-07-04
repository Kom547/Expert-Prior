# P0 实验计划：Cross-Victim Prior Transfer

## 目标审稿意见

- Reviewer 1-C4：Table 3 中每个 AEGRL cell 使用 in-distribution victim demonstrations，不能证明 expert prior 能跨 victim policy transfer。

## 要验证的 claim

由某一 victim policy 生成的 demonstrations 训练得到的 expert prior，是否能有效指导 AEGRL 攻击另一 victim policy。

## 实验设置

- 环境：Env-1，`TrafficEnv3-v5`。
- Source victim：PPO、SAC、TD3。
- Target victim：PPO、SAC、TD3。
- 形成 3x3 transfer matrix。
- 默认 attack setting：`eps=0.05`、`Gamma=4`、`adv_steps=4`。
- Seeds：0, 1, 2, 3, 4。
- Evaluation episodes：100。
- 主指标：CR、AE/ANA、attack times。

## 代码入口

- Demo 采集：`evaluation_v3.py --expert_recording`
- Expert prior 训练：`expert_imitation_learning_MoE.py`
- AEGRL adversary 训练：`advTrain.py --expert_attack --expert_model_path <source_prior>`
- Cross-target 评估：`evaluation_v3.py --algo <target_victim> --expbase_algo <source_or_train_base>`
- 可参考脚本：`Scripts/generation_test.sh`

## 推荐命名规范

- Expert data：`expert_data/transfer_src_<SRC>_env3_eps0.05/`
- Expert model：`expert_model/transfer/env3_src_<SRC>/`
- AEGRL run：`transfer_env3_src<SRC>_tgt<TGT>_seed<SEED>`
- Result CSV：`evaluation_result/transfer_env3_3x3.csv`

## 输出表/图

- 主表：3x3 matrix，每个 cell 报告 `CR mean ± std` 和 `AE mean ± std`。
- 辅助表：off-diagonal 相对 diagonal 的 performance retention ratio。
- 可选图：heatmap，横轴 target victim，纵轴 source victim。

## 统计方法

- 对每个 target victim，比较 diagonal source 与 off-diagonal source 的 CR/AE。
- 报告 mean、95% CI。
- 若需要显著性，使用 paired t-test 或 Wilcoxon，Holm correction。

## 验收标准

- 9 个 source-target cell 均有 5 seeds 结果。
- 对每个 target victim 至少能回答：跨 victim prior 是否保持有效、是否存在明显 victim-specific prior。
- 如果 off-diagonal 明显下降，正文必须将 claim 改为“可在一定质量阈值或相近 victim dynamics 下 transfer”。

## 论文对应修改位置

- Section 5.5 或新增 Section 5.6：Cross-victim transfer study。
- Introduction/Conclusion：按结果调整 generality claim。
- Response letter：直接回应 Reviewer 1-C4。

## Rebuttal 回应句

我们新增了 Env-1 的 3x3 source-target victim transfer 实验，用 PPO/SAC/TD3 demonstrations 分别训练 expert prior，再评估其对 PPO/SAC/TD3 victims 的 AEGRL 指导效果。

