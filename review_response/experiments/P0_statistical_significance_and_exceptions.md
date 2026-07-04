# P0 实验计划：统计显著性与例外结果解释

## 目标审稿意见

- Reviewer 2：缺少统计显著性分析，且 TD3 Env-2、PPO Env-2 存在与“best in almost all cases”不一致的例外。
- Reviewer 3-C7：结果标准差较高，缺少 hypothesis tests。

## 要验证的 claim

AEGRL 在多数设置下相对 Vanilla、VPRL、PCRL 具有统计可靠的 CR/AE 改进；对不显著或反向的设置，论文必须明确承认并解释原因。

## 实验设置

- 数据来源：现有 5 seeds 的 evaluation CSV 或重新运行主结果评估。
- 默认环境：`TrafficEnv3-v5`、`TrafficEnv8-v1`。
- Victim：PPO、SAC、TD3。
- 方法：Vanilla、VPRL、PCRL、AEGRL；若 expert-only baseline 有 seed-level 结果，也一并统计。
- 指标：CR、AE/ANA、episode length、attack times。
- 每个 trained model：100 evaluation episodes。

## 代码入口

- 评估入口：`evaluation_v3.py`
- 结果汇总输入：`evaluation_result/*.csv`
- 训练日志辅助：`logs/adv_eval/**/rollout_log.csv`
- 需要新增的后续分析脚本建议命名：`review_response/scripts/analyze_significance.py`

## 统计方法

1. 对每个 env/victim/method 聚合 5 seeds 的 mean 和 95% CI。
2. AEGRL vs 每个 baseline：
   - 若 seed 一一对应，优先 paired t-test。
   - 若 Shapiro-Wilk 正态性不满足，使用 Wilcoxon signed-rank test。
3. 对同一表内多重比较使用 Holm correction。
4. 报告效应量：
   - paired t-test 使用 Cohen's dz。
   - Wilcoxon 使用 rank-biserial correlation。
5. 对 PPO Env-2 和 TD3 Env-2 输出单独诊断表，列出 seed-level CR/AE、std、CI、显著性与 possible cause。

## 输出表/图

- `review_response/results/statistical_significance_seed_level.csv`
- `review_response/results/statistical_significance_summary.csv`
- `review_response/figures/significance_forest_plot.pdf`
- 论文新增或替换表：主结果表增加 CI/p-value 标记，或新增 Appendix 表。

## 验收标准

- 每个主实验设置都有 seed-level 记录。
- 每个 AEGRL 与 baseline 对比都有 p-value、adjusted p-value、effect size。
- 明确标注不显著和反向结果，不能继续使用“almost all cases”这类过强措辞。

## 论文对应修改位置

- Section 5.5.2：改写 performance evaluation 结论。
- Table 3 或 Appendix：加入 CI 和统计检验。
- Response letter：说明已补充统计检验，并承认 Env-2 特殊结果。

## Rebuttal 回应句

我们补充了基于 5 个随机种子的置信区间、配对显著性检验和多重比较校正，并在修订稿中明确讨论了 Env-2 中 AEGRL 不占优或方差较大的例外设置。

