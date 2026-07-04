# P1 实验计划：超参数敏感性分析

## 目标审稿意见

- Reviewer 3-C4：MoE architecture choices 和 annealing parameters 缺少 justification 和 sensitivity analysis。

## 要验证的 claim

AEGRL 的性能对关键超参数不应只依赖单个偶然设置；默认配置应位于稳定区间或性能-成本折中合理的位置。

## 实验设置

优先用 Env-1 TD3 victim 或 Env-1 PPO victim 做单场景敏感性，再抽查 Env-2。

- 环境：优先 `TrafficEnv3-v5`。
- Victim：TD3 或 PPO；若计算资源允许，扩展 SAC。
- Seeds：0..4；若成本过高，先 3 seeds 做趋势，再补齐 5 seeds。
- Evaluation episodes：100。

## 超参数维度

1. Annealing exponent `k`：
   - 候选：0.5、2.0、3.0、4.0、5.0、6.5。
   - 复用：`Scripts/kTest.sh`。
2. Beta/annealing mechanism：
   - Full AEGRL。
   - `Fix` fixed coefficient。
   - `wo_beta` no adaptive beta。
3. MoE ensemble count：
   - 1、3、5。
   - 当前 `load_ensemble_models` 默认读取 ensemble_1 到 ensemble_5，可通过保存不同数量模型或加载子集实现。
4. Experts per MoE：
   - 1、3、5。
   - 需要后续实现给 `Actor(num_experts=...)` 暴露 config。
5. Sampling ratio / data quality：
   - 0.5、0.3、0.1、0.05、0.01。
   - 复用：`Scripts/MoE_experiment.sh`。

## 代码入口

- Training：`advTrain.py --expert_k <k> --wo_beta`
- Expert training：`expert_imitation_learning_MoE.py --sampling_ratio <ratio>`
- Existing scripts：`Scripts/kTest.sh`、`Scripts/MoE_experiment.sh`
- Evaluation：`evaluation_v3.py`

## 输出表/图

- Line plot：`k` vs CR/AE。
- Bar chart：Full vs Fix vs wo_beta。
- Table：MoE count / expert count vs CR/AE / parameter count。
- Data ratio sensitivity plot：sampling ratio vs expert-only CR and AEGRL CR。

## 统计方法

- 每个 setting 汇总 mean ± std、95% CI。
- 对 default setting 与其他 settings 做 paired test。
- 对趋势使用 Spearman correlation 或 simple regression。

## 验收标准

- 至少完成 `k` sensitivity 和 beta mechanism sensitivity。
- MoE architecture 至少完成 ensemble count sensitivity；experts per MoE 若代码改动过大，可作为 Appendix/limitation。
- 正文给出默认配置选择理由：性能稳定、成本适中、不过度依赖单点参数。

## 论文对应修改位置

- Section 5.3 Hyperparameter Settings：补充选择依据。
- Section 5.7 Ablation 或 Appendix：新增 sensitivity results。
- Response letter：回应关键超参数缺少 justification 的问题。

## Rebuttal 回应句

我们补充了 annealing 参数和 MoE 架构的敏感性分析，说明默认配置并非单点调参结果，而是在性能稳定性和计算成本之间取得的折中。

