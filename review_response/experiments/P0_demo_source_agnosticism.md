# P0 实验计划：Demo Source Agnosticism

## 目标审稿意见

- Reviewer 1-C1：手稿声称 expert generation module 对 demonstration source agnostic，但所有实验都来自 [14]。
- Reviewer 3-C6：需要展示不同数据源训练出的 expert policies 是否能带来相似或更好的最终性能。

## 要验证的 claim

AEGRL 的 expert generation module 可以从不同 demonstration sources 中提取可用 prior，但性能会受到 source quality 和 coverage 的影响。

## 实验设置

最小闭环优先在 Env-1 PPO victim 上完成，若结果稳定再扩展 SAC/TD3。

- 环境：`TrafficEnv3-v5`。
- Victim：PPO；扩展项为 SAC、TD3。
- Demo sources：
  1. `[14] relaxed-budget sequential attacker`：当前默认 source。
  2. `heuristic random-time random-direction`：随机选择 attack timing，随机选择 target action 或 perturbation direction。
  3. `weak myopic/FGSM source`：使用较弱的 myopic FGSM 或低训练步数 sequential attacker。
- 每类 source 采集相同规模 demonstration data。
- 每类 source 独立训练 MoE expert prior。
- 使用每个 prior 独立训练 AEGRL adversary。
- Seeds：0..4。
- Evaluation episodes：100。
- 指标：expert-only CR/AE、AEGRL CR/AE、BC imitation loss、demo success rate、state coverage。

## 代码入口

- 当前默认 demo 采集：`evaluation_v3.py --expert_recording`
- Heuristic source：后续实现可新增一个轻量 demo generator，或在 `evaluation_v4.py` 的 action generation 前替换 adv action。
- Weak FGSM source：优先复用 `st_attack_fgsm.py` 或 `st_attack_fgsm_exp_alternation_vec.py` 的低预算/低训练步设置。
- Expert training：`expert_imitation_learning_MoE.py`
- AEGRL training：`advTrain.py`
- Evaluation：`evaluation_v3.py`

## 输出表/图

- 表 1：各 demo source 的数据质量统计，包含 trajectories、successful episodes、state coverage、mean attack times。
- 表 2：各 source 对应 expert-only 和 AEGRL 的 CR/AE。
- 图：source quality vs AEGRL final CR scatter plot。

## 统计方法

- 每个 source 使用 5 seeds 汇总 mean ± std 和 95% CI。
- AEGRL([14]) vs AEGRL(heuristic/weak) 做 paired 或 unpaired test，取决于是否共享 seed pairing。
- 报告 source quality 与 final CR 的 Spearman correlation。

## 验收标准

- 至少完成两个 contrasting sources：默认 [14] 和 heuristic source。
- 能明确支持以下二选一结论：
  - 若差距小：支持 agnostic claim。
  - 若差距大：将 claim 降调为“module can ingest different sources, while final performance depends on source quality”。

## 论文对应修改位置

- Section 4.2.1：将 “agnostic” 改为更精确的可操作表述。
- Section 5.6 或新增 Appendix：补充 demo source study。
- Limitation：讨论 demonstration source quality threshold。

## Rebuttal 回应句

我们新增了不同 demonstration source 的对比实验，显示 AEGRL 的 expert module 可以接收不同来源的 adversarial trajectories，同时最终性能与 source quality 和 coverage 密切相关。

