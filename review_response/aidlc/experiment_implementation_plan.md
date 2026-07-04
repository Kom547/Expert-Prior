# AEGRL 补充实验实现计划

生成日期：2026-06-13  
目的：给后续 coding agent 使用的实现计划。只关注实验代码、脚本、命令和原始结果产物。

## 1. Non-Goals

本计划不做以下事情：

- 不修改论文正文。
- 不写 response letter。
- 不判断实验结果是否好。
- 不做显著性检验或结果解释。
- 不绘制最终论文图。

## 2. Target Files to Add Later

建议后续实现新增这些脚本，而不是把逻辑塞进现有训练文件：

```text
review_response/scripts/
  collect_main_raw_results.py
  generate_experiment_commands.py
  run_command_manifest.py
  collect_moe_router_outputs.py
  generate_heuristic_demo_source.py
  profile_experiment_run.py
```

建议新增配置：

```text
review_response/configs/
  exp001_cross_victim_transfer.yaml
  exp002_demo_source_agnosticism.yaml
  exp003_moe_router_collection.yaml
  exp004_main_result_completion.yaml
  exp005_hyperparameter_sweep.yaml
  exp006_profiling.yaml
  exp007_sac_backbone_feasibility.yaml
  exp008_third_scenario_feasibility.yaml
```

建议统一输出：

```text
review_response/experiment_runs/
  commands/
  logs/
  raw_results/
  metadata/
  blockers/
```

## 3. Implementation Units

### Unit 1：统一实验命令与结果 schema

对应需求：EXP-004 的基础设施。

实现内容：

1. 新增 `generate_experiment_commands.py`：
   - 输入 YAML config。
   - 输出 `.ps1` 或 `.sh` 命令清单。
   - 每条命令包含 run id。
2. 新增 `run_command_manifest.py`：
   - 顺序或并行运行命令。
   - 记录 start/end time。
   - 捕获 stdout/stderr 到 `logs/`。
3. 新增 raw result schema 写入工具：
   - 标准化 `evaluation_v3.py` 的输出字段。
   - 追加到 `raw_results/<experiment_id>.csv`。

验收：

- 能为一个简单 config 生成 2 条命令。
- 命令执行后能生成 log 和 raw CSV。

### Unit 2：主实验结果补齐扫描

对应需求：EXP-004。

实现内容：

1. 新增 `collect_main_raw_results.py`。
2. 扫描：
   - `evaluation_result/*.csv`
   - `logs/adv_eval/**/rollout_log.csv`
3. 规范化已有结果到：
   - `review_response/experiment_runs/raw_results/EXP004_main_results.csv`
4. 输出缺失矩阵：
   - env x victim x method x seed。
5. 生成缺失补跑命令。

验收：

- 输出一个完整/缺失状态表。
- 不要求自动补跑；只要能生成补跑命令。

### Unit 3：MoE router raw data 采集

对应需求：EXP-003。

实现内容：

1. 新增 `collect_moe_router_outputs.py`。
2. 参数：
   - `--expert_model_path`
   - `--input_data_path`
   - `--output_csv`
   - `--device`
   - `--model_count`
3. 加载 `expert_imitation_learning_MoE.Actor`。
4. 对输入 states 输出：
   - router weights
   - dominant expert
   - entropy
   - optional expert output means/stds

验收：

- 对一个 `.npz` 文件能成功输出 per-sample CSV。
- 不做 heatmap，不做结果解释。

### Unit 4：Cross-victim transfer 命令生成

对应需求：EXP-001。

实现内容：

1. 配置文件定义：
   - source victims
   - target victims
   - seeds
   - env
   - eps
   - adv_steps
2. 命令生成分三阶段：
   - demo collection commands。
   - MoE expert training commands。
   - AEGRL train/eval commands。
3. 输出：
   - `commands/EXP001_collect_demo.*`
   - `commands/EXP001_train_prior.*`
   - `commands/EXP001_train_eval_aegrl.*`

验收：

- 能生成 3 x 3 x 5 seeds 的 train/eval 命令。
- 命令中 run id 可追踪 source 和 target。

### Unit 5：Heuristic demo source generator

对应需求：EXP-002。

实现内容：

1. 新增 `generate_heuristic_demo_source.py`。
2. 复用 environment 和 victim loading 逻辑。
3. 在每个 episode 中：
   - 随机选择 attack timestep 或按概率触发。
   - 生成随机 target action / perturb direction。
   - 保存 `obs` 和 `adv_actions`，格式兼容 `expert_imitation_learning_MoE.py`。
4. 输出 source metadata：
   - episodes
   - saved samples
   - collision count if available
   - env/victim/eps/adv_steps/seed

验收：

- 生成的目录能被 `expert_imitation_learning_MoE.py --expert_data_path` 读取。
- 至少一个 PPO/Env-1 demo source 可成功训练 MoE prior。

### Unit 6：Demo source agnosticism 命令生成

对应需求：EXP-002。

实现内容：

1. 为 source A/B/C 生成：
   - demo generation commands。
   - prior training commands。
   - AEGRL training commands。
   - evaluation commands。
2. 最小版本只跑：
   - Env `TrafficEnv3-v5`
   - Victim PPO
   - Source `vanilla_14_relaxed`
   - Source `heuristic_random_time_direction`

验收：

- 至少两类 source 的完整命令链可生成。

### Unit 7：Hyperparameter sweep 命令生成

对应需求：EXP-005。

实现内容：

1. 将 `expert_k` sweep 写入 YAML。
2. 支持 beta mode：
   - full
   - fixed
   - wo_beta
3. 生成 train/eval 命令。
4. 输出 raw CSV。

验收：

- 能生成所有 `k` x seed 的命令。
- 每条命令 run id 包含 `k` 和 beta mode。

### Unit 8：Profiling wrapper

对应需求：EXP-006。

实现内容：

1. 新增 `profile_experiment_run.py` 或给 `run_command_manifest.py` 增加 profiling。
2. 记录：
   - wall-clock start/end。
   - exit code。
   - command。
   - model parameter count if model path exists。
   - hardware metadata。
3. 输出：
   - `raw_results/EXP006_profiling_raw.csv`

验收：

- 任意命令运行后都有 profiling row。

### Unit 9：Third scenario feasibility checker

对应需求：EXP-008。

实现内容：

1. 检查 `TrafficEnv7-v0`：
   - gym make。
   - reset。
   - step。
   - attack=True。
2. 检查 victim model path。
3. 检查 observation shape 是否符合 attack/expert pipeline。
4. 生成 feasibility report。

验收：

- 如果通过，生成 Vanilla/AEGRL 最小命令。
- 如果失败，生成 blocker report。

### Unit 10：SAC adversary feasibility checker

对应需求：EXP-007。

实现内容：

1. 检查当前 `advTrain.py --adv_algo SAC` 是否只是 evaluation load，还是能训练 SAC adversary。
2. 如果不能训练，输出 blocker report 和需要新增的类/方法。
3. 如果能训练，生成 Env-1/PPO victim 的最小 SAC-Vanilla、SAC-AEGRL 命令。

验收：

- 不强制实现完整 SAC-AEGRL。
- 必须产出可执行命令或 blocker report。

## 4. Implementation Order

推荐顺序：

1. Unit 1：统一命令与结果 schema。
2. Unit 2：主实验结果补齐扫描。
3. Unit 3：MoE router raw data。
4. Unit 4：Cross-victim transfer 命令生成。
5. Unit 5：Heuristic demo generator。
6. Unit 6：Demo source agnosticism 命令生成。
7. Unit 7：Hyperparameter sweep。
8. Unit 8：Profiling wrapper。
9. Unit 9：Third scenario feasibility。
10. Unit 10：SAC adversary feasibility。

## 5. First Agent Task Recommendation

第一个实现 agent 应该从 Unit 1 + Unit 2 开始：

- 它们不需要改训练算法。
- 能先摸清已有结果缺口。
- 后续所有实验都依赖统一 run id、命令记录和 raw CSV schema。

完成 Unit 1 + Unit 2 后，再启动 Unit 3，因为 MoE router raw data 是低成本新增实验产物。

## 6. Done Definition

本实现计划完成的标准不是“论文能写了”，而是：

- 所有 EXP-001 到 EXP-008 都有脚本入口或 blocker report。
- 所有可运行实验都有命令清单。
- 所有 evaluation 结果能进入统一 raw CSV。
- 后续 agent 不需要再判断“该跑什么”，只需要按 config 和命令执行。

