import pandas as pd
import numpy as np
import re
#处理攻击性能测试文件，分组计算
# 加载原始 CSV 文件
df = pd.read_csv("evaluation_result/AttackEvaluation_0612.csv")

# 提取关键信息
def parse_adv_agent(agent_str):
    parts = agent_str.split('_')
    seed = int(parts[-1]) if parts[-1].isdigit() else None
    group_keywords = ['nobeta', 'nlg', 'single', 'k6.5']
    group = next((kw for kw in group_keywords if kw in agent_str), 'control')
    as_match = re.search(r'as(\d+)', agent_str)
    attack_steps = int(as_match.group(1)) if as_match else None
    method_match = next((m for m in ['ValuePenalty', 'PolicyConstrained', 'Vanilla'] if m in agent_str), None)
    normalized_agent = '_'.join(parts[:-1]) if seed is not None else agent_str
    return pd.Series([group, attack_steps, method_match, seed, normalized_agent])

df[['experiment_group', 'attack_steps', 'attack_method', 'seed', 'normalized_agent']] = df['adv_agent'].apply(parse_adv_agent)

# 聚合函数，保留元数据
def compute_stats(group):
    n = len(group)
    return pd.Series({
        'env_name': group['env_name'].iloc[0],
        'algo': group['algo'].iloc[0],
        'attack_eps': group['attack_eps'].iloc[0],
        'mean_attack_times_mean': group['mean_attack_times'].mean(),
        'mean_attack_times_ci': 1.96 * group['mean_attack_times'].std(ddof=1) / np.sqrt(n) if n > 1 else 0,
        'collision_rate_mean': group['collision_rate'].mean(),
        'collision_rate_ci': 1.96 * group['collision_rate'].std(ddof=1) / np.sqrt(n) if n > 1 else 0
    })

# 聚合
result = df.groupby(['normalized_agent', 'experiment_group', 'attack_steps', 'attack_method']).apply(compute_stats).reset_index()
#result = df.apply(compute_stats).reset_index()
# 保存结果
result.to_csv("evaluation_result/Processed_AttackEvaluation_0612.csv", index=False)
