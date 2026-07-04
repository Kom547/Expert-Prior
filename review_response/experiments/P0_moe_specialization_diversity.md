# P0 实验计划：MoE 专门化与攻击多样性分析

## 目标审稿意见

- Reviewer 1-C3：没有证据表明 MoE experts 真的 specialize。
- Reviewer 3-C8：Collision rate 不能反映攻击模式多样性。

## 要验证的 claim

Ensemble MoE 的优势不仅来自参数量或 ensemble averaging，还体现在不同 experts/router 对不同 traffic states、attack phases 和 attack/idle decisions 的差异化响应。

## 实验设置

- 使用已有 MoE expert models，优先路径：`expert_model/MoEs/f_0.5` 或主实验实际使用模型。
- 数据：
  - Demonstration states。
  - AEGRL evaluation rollouts。
  - Env-1 和 Env-2 都应覆盖。
- 分组维度：
  - Env：`TrafficEnv3-v5` vs `TrafficEnv8-v1`。
  - Victim：PPO/SAC/TD3。
  - State phase：intersection/merge critical region、straight/approach region；可用 x/y position 或 timestep proxy。
  - Attack status：attack step vs idle step。

## 代码入口

- MoE model definition：`expert_imitation_learning_MoE.py` 中 `Actor.gating_network`。
- Demo 数据读取：`load_expert_samples` 读取 `.npz`。
- Evaluation trajectory 保存：可复用 `evaluation_csv.py` 或 `evaluation_v4.py --save_samples` 逻辑。
- 后续分析脚本建议：`review_response/scripts/analyze_moe_specialization.py`

## 指标定义

- Router activation mean：每组状态下 3 个 experts 的平均 softmax 权重。
- Dominant expert ratio：`argmax(router weights)` 的占比。
- Router entropy：`-sum p_i log p_i`，越低表示更明确的 expert selection。
- Inter-expert output distance：不同 experts 输出 mean action 的平均 L2 distance。
- Attack timing diversity：攻击发生 timestep 的 entropy 或 histogram。
- State coverage：攻击状态在关键状态特征上的 cluster count 或 convex hull proxy。

## 输出表/图

- Router activation heatmap：行是状态组，列是 expert id。
- Dominant expert bar chart：不同 env/victim/phase 的 expert selection 占比。
- Router entropy table：Env-1、Env-2、attack、idle 对比。
- Attack timing histogram：S-MLP、S-MoE、Ens-MoE、AEGRL 对比。

## 统计方法

- 对每个 group 计算 mean ± bootstrap CI。
- 对 router entropy 和 inter-expert distance 可用 Mann-Whitney U 或 Kruskal-Wallis test 比较分组差异。

## 验收标准

- 至少输出一张 heatmap 和一张 entropy/dominant expert 表。
- 能说明 experts 是否存在可解释的分工。
- 如果专门化不明显，则正文应弱化“specialized subnetworks”叙述，改为“mixture/ensemble improves robustness and reduces model bias”。

## 论文对应修改位置

- Section 5.5.1：补充 expert policy diversity/specialization analysis。
- Figure：新增 router activation heatmap。
- Response letter：回应 MoE 专门化与 CR 不等于 diversity 的质疑。

## Rebuttal 回应句

我们新增了 router activation、dominant expert ratio 和 router entropy 分析，以直接展示 MoE prior 在不同场景和攻击阶段中的 expert selection 差异，而不仅依赖 collision rate。

