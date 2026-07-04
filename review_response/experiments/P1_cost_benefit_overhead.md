# P1 实验计划：成本收益与复杂度分析

## 目标审稿意见

- Reviewer 3-C9：报告提升相对 modest，但没有说明额外计算成本是否值得。
- Reviewer 3-C2：方法是已有技术组合，需要证明组合后的实际收益。

## 要验证的 claim

AEGRL 相比 Vanilla ([14])、VPRL、PCRL 的额外成本是可接受的，并且在 failure discovery effectiveness 上带来合理收益。

## 实验设置

- 环境：`TrafficEnv3-v5`、`TrafficEnv8-v1`。
- Victim：PPO、SAC、TD3。
- Methods：Vanilla ([14])、VPRL、PCRL、AEGRL。
- Seeds：至少 3 seeds 记录成本，最好 5 seeds。
- Evaluation episodes：100。

## 成本指标

- Expert training time：训练 MoE prior 的 wall-clock time。
- Adversary training time：训练 attack policy 的 wall-clock time。
- Evaluation time：100 episodes evaluation 的 wall-clock time。
- Inference latency：单步 action selection 平均耗时。
- Parameter count：expert prior 参数量、adversary policy 参数量。
- Model count：5 MoE x 3 experts 的结构开销。
- GPU/CPU 环境：记录 GPU 型号、CUDA、PyTorch、Stable-Baselines3 版本。

## 收益指标

- CR absolute gain：AEGRL CR - baseline CR。
- CR relative gain。
- AE/ANA gain。
- Failure discovery efficiency：CR / mean attack times 或 CR / training hour。
- Negative transfer mitigation：Env-2 中 AEGRL 相对固定 expert-guided baseline 的 gain。

## 代码入口

- Training：`advTrain.py`
- Expert training：`expert_imitation_learning_MoE.py`
- Evaluation：`evaluation_v3.py`
- 参数量统计：读取 PyTorch model 后 `sum(p.numel() for p in model.parameters())`
- 延迟统计：后续新增轻量 profiling 脚本，建议 `review_response/scripts/profile_overhead.py`

## 输出表/图

- Cost-benefit table：每个 method 的 CR/AE、训练时间、推理延迟、参数量。
- Scatter plot：x 轴 training time，y 轴 CR。
- Bar chart：relative gain vs overhead ratio。

## 统计方法

- 成本指标报告 mean ± std。
- 由于运行环境可能变化，成本表应说明硬件配置。
- 不做过度显著性解释，重点是 practical trade-off。

## 验收标准

- 至少报告 AEGRL、Vanilla ([14])、VPRL/PCRL 的训练时间、评估时间和参数量。
- 正文必须明确解释 modest improvement 在 safety-critical evaluation 中的意义，且不夸大。
- 如果 overhead 很高，应在 limitation 中承认，并强调 AEGRL 用于离线 robustness evaluation，不是实时部署模块。

## 论文对应修改位置

- Section 5.5 或新增 Practical Analysis。
- Conclusion/Limitation：说明计算成本与适用场景。
- Response letter：回应 practical significance 和 added complexity。

## Rebuttal 回应句

我们新增了成本收益分析，报告训练时间、推理延迟、参数量以及相对 CR/AE 收益，以说明 AEGRL 的额外复杂度在离线鲁棒性评估场景下是可接受的。
