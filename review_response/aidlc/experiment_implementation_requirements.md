# AEGRL 补充实验实现需求文档

生成日期：2026-06-13  
目的：只提取需要新增或补齐的实验实现任务，供后续 coding agent 按任务实现脚本、配置和运行入口。  
非目标：本文档不规划论文修改、不写 response letter、不解释实验结果、不做结果显著性分析。

## 1. Scope

本阶段只关注“需要让代码能跑起来并产出原始实验结果”的工作，包括：

- 生成或复用实验数据。
- 训练 expert prior 或 adversary。
- 运行 evaluation。
- 保存 seed-level 原始结果、命令、日志和模型路径。
- 为后续人工或单独分析脚本保留足够数据。

不包括：

- 修改论文正文。
- 组织 response letter。
- 分析结果是否支持 claim。
- 统计显著性解释。
- 绘制 manuscript-ready 图表。

## 2. Implementation Requirements

| ID | 实验实现需求 | 优先级 | 需要实现什么 | 完成判定 |
|---|---|---:|---|---|
| EXP-001 | Cross-victim prior transfer 实验 | P0 | 生成 source-victim specific expert prior，并在 target victim 上训练/评估 AEGRL。 | 能跑出 Env-1 的 PPO/SAC/TD3 source x PPO/SAC/TD3 target 原始结果。 |
| EXP-002 | Demo source agnosticism 实验 | P0 | 增加不同 demonstration source 的数据生成、expert prior 训练和 AEGRL 评估流程。 | 至少能跑 `[14]-derived/Vanilla source` 与 heuristic source 两类 demo source。 |
| EXP-003 | MoE router/specialization 数据采集 | P0 | 实现脚本读取 MoE `gating_network`，对 trajectory states 输出 router weights 和 expert usage raw data。 | 生成每条样本的 router weights、dominant expert、entropy、env/victim/method metadata。 |
| EXP-004 | 主实验 seed-level raw result 补齐 | P0 | 确保现有 Vanilla/VPRL/PCRL/AEGRL 主实验每个 env/victim/method/seed 都有可机器读取结果。 | 生成统一 schema 的 seed-level CSV；不要求做统计分析。 |
| EXP-005 | Hyperparameter sweep 实验 | P1 | 将 `k`、fixed/no-beta、MoE sampling ratio 等 sweep 整理成可重复运行脚本。 | 每个 sweep setting 有独立 run id、日志和 seed-level result。 |
| EXP-006 | Cost/profiling raw data 采集 | P1 | 在训练/评估入口记录 wall-clock time、参数量、evaluation time。 | 输出 profiling raw CSV；不需要解释成本收益。 |
| EXP-007 | SAC adversary backbone feasibility | P2 | 尝试新增或封装 SAC adversary 训练入口。 | 最小 Env-1/PPO victim 配置能完成训练和 evaluation，或记录不可实现原因。 |
| EXP-008 | 第三场景 feasibility 与最小实验 | P2 | 检查 `TrafficEnv7-v0` 是否有 victim model 和 attack 兼容性，并跑 AEGRL vs Vanilla 最小实验。 | 若兼容，生成至少 PPO victim 的 raw result；若不兼容，输出 blocker report。 |

## 3. Common Experiment Contract

所有新增实验实现必须遵守统一产物约定。

### 3.1 Run ID

建议格式：

```text
<experiment_id>__env=<env>__victim=<victim>__source=<source>__method=<method>__eps=<eps>__steps=<adv_steps>__seed=<seed>
```

示例：

```text
EXP001__env=TrafficEnv3-v5__victim=TD3__source=PPO__method=AEGRL__eps=0.05__steps=4__seed=0
```

### 3.2 Required Output Directories

```text
review_response/experiment_runs/
  commands/
  logs/
  raw_results/
  metadata/
  blockers/
```

### 3.3 Required CSV Schema

每个 evaluation run 至少输出：

| Field | Description |
|---|---|
| `experiment_id` | EXP-001 等任务编号。 |
| `run_id` | 唯一运行名。 |
| `env_name` | Gym/SUMO 环境名。 |
| `victim_algo` | PPO/SAC/TD3。 |
| `source_type` | prior/demo 来源；无 source 时填 `none`。 |
| `source_victim` | PPO/SAC/TD3/none。 |
| `method` | Vanilla、AEGRL、VPRL、PCRL、ExpertOnly 等。 |
| `seed` | 随机种子。 |
| `attack_eps` | 扰动幅度。 |
| `adv_steps` | 攻击预算。 |
| `train_steps` | 训练步数配置。 |
| `eval_episodes` | evaluation episode 数。 |
| `collision_rate` | evaluation 输出的 mean reward / CR。 |
| `mean_attack_times` | 平均攻击次数。 |
| `model_path` | 被评估 adversary model 路径。 |
| `expert_model_path` | 使用的 expert prior 路径；无则空。 |
| `command` | 完整命令。 |
| `status` | success/failed/skipped。 |

### 3.4 Blocker Report

如果某个实验无法运行，agent 必须输出：

```text
review_response/experiment_runs/blockers/<run_id>.md
```

内容包含：

- 失败命令。
- stderr 或关键报错。
- 缺失文件/模型路径。
- 是否需要代码修改。
- 建议下一步。

## 4. Experiment-Specific Requirements

### EXP-001 Cross-victim prior transfer

目标：实现 source victim 到 target victim 的 prior transfer 实验运行链路。

最小配置：

- Env：`TrafficEnv3-v5`
- Source victim：PPO、SAC、TD3
- Target victim：PPO、SAC、TD3
- Method：AEGRL
- Seeds：0..4
- `attack_eps=0.05`
- `adv_steps=4`
- `eval_episodes=100`

实现任务：

1. 为每个 source victim 准备 expert data 路径。
2. 为每个 source victim 训练 MoE expert prior。
3. 使用 source-specific prior 训练 AEGRL adversary。
4. 在每个 target victim 上 evaluation。
5. 保存 9 cells x 5 seeds 的 raw CSV。

### EXP-002 Demo source agnosticism

目标：实现不同 demonstration source 的数据生成与 AEGRL 训练链路。

最小配置：

- Env：`TrafficEnv3-v5`
- Victim：PPO
- Source A：`vanilla_14_relaxed`
- Source B：`heuristic_random_time_direction`
- Optional Source C：`weak_fgsm_or_weak_sequential`
- Seeds：0..4

实现任务：

1. 复用现有 `[14]-derived/Vanilla` demo collection。
2. 新增 heuristic demo generator：
   - 随机 attack timing。
   - 随机 target action 或 perturbation direction。
   - 保存与现有 `.npz` expert data 兼容的 `obs` 和 `adv_actions`。
3. 可选新增 weak FGSM/sequential source。
4. 每类 source 训练 MoE prior。
5. 每类 source 训练/评估 AEGRL。
6. 保存 source metadata，包括 demo 数量、成功样本数量、trajectory 数量。

### EXP-003 MoE router/specialization raw data

目标：实现 MoE router 输出数据采集，不负责图表解释。

输入：

- MoE model path，例如 `expert_model/MoEs/f_0.5`
- Expert data 或 evaluation trajectory `.npz`/CSV

实现任务：

1. 加载 `expert_imitation_learning_MoE.Actor`。
2. 对每条 state 计算 `F.softmax(gating_network(state))`。
3. 输出 per-sample raw CSV：
   - sample id
   - env
   - victim
   - method
   - timestep
   - attack flag
   - router weight 0/1/2
   - dominant expert
   - entropy
4. 不要求绘图。

### EXP-004 主实验 raw result 补齐

目标：把已有/补跑的主实验结果整理成统一 schema，供后续分析使用。

范围：

- Env：`TrafficEnv3-v5`、`TrafficEnv8-v1`
- Victim：PPO、SAC、TD3
- Methods：Vanilla、VPRL、PCRL、AEGRL
- Seeds：0..4

实现任务：

1. 扫描已有 `evaluation_result/*.csv`。
2. 扫描已有 `logs/adv_eval/**`。
3. 识别缺失 cell。
4. 为缺失 cell 生成待运行命令文件。
5. 可选执行补跑。
6. 输出统一 raw CSV。

### EXP-005 Hyperparameter sweep

目标：实现可重复运行的超参数 sweep 命令生成与结果采集。

最小 sweep：

- `expert_k`: 0.5、2.0、3.0、4.0、5.0、6.5
- beta mode：full、fixed、wo_beta
- Env：`TrafficEnv3-v5`
- Victim：优先 TD3 或 PPO
- Seeds：0..4

实现任务：

1. 整理 `Scripts/kTest.sh` 为可配置脚本或命令生成器。
2. 对每个 setting 生成 run id。
3. 保存完整命令。
4. evaluation 后写统一 raw CSV。

### EXP-006 Cost/profiling raw data

目标：记录实验运行成本原始数据，不做成本收益解释。

实现任务：

1. 在命令 wrapper 中记录 start/end wall-clock time。
2. 统计模型参数量。
3. 记录 hardware metadata：device、CUDA availability、Python/PyTorch/SB3 version。
4. 输出 `profiling_raw.csv`。

### EXP-007 SAC adversary backbone feasibility

目标：实现或验证 SAC adversary AEGRL 的最小可行性。

最小配置：

- Env：`TrafficEnv3-v5`
- Victim：PPO
- Methods：SAC-Vanilla、SAC-AEGRL
- Seeds：0..2 first pass

实现任务：

1. 检查当前 `advTrain.py` 是否能真正训练 SAC adversary。
2. 如果不能，评估需要新增的 SAC adversarial algorithm wrapper。
3. 若实现，保存最小 raw result。
4. 若不可行，输出 blocker report，不继续消耗时间。

### EXP-008 第三场景 feasibility 与最小实验

目标：验证 `TrafficEnv7-v0` 是否能用于补充实验。

实现任务：

1. 检查 env 是否可创建并 reset/step。
2. 检查 victim model 是否存在。
3. 检查 attack mode 下 observation/action shape 是否兼容。
4. 若通过，跑 Vanilla 与 AEGRL 最小实验。
5. 若失败，输出 blocker report。

## 5. Priority

执行顺序：

1. EXP-004：先知道已有结果缺什么。
2. EXP-003：MoE router raw data，低成本。
3. EXP-001：Cross-victim transfer，核心新增实验。
4. EXP-002：Demo source agnosticism，核心但成本更高。
5. EXP-005：Hyperparameter sweep。
6. EXP-006：Profiling raw data。
7. EXP-008：第三场景 feasibility。
8. EXP-007：SAC adversary feasibility。

