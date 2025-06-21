import os
import re
import argparse
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def collect_and_plot_eval(root_dir: str,
                     keyword: str = 'ab_ens2',
                     clean: bool = False):
    # Pattern to extract metadata from directory names
    pattern = re.compile(
        r'^(?P<prefix>.*?)_(?P<env>TrafficEnv\d+-v\d+)_(?P<alg>\w+)_(?P<attack>[^_]+)_eps(?P<eps>[\d.]+)_as(?P<as>\d+)_(?P<seed>\d+)$'
    )

    # Prepare output directory
    output_dir = os.path.join(root_dir, 'plots')
    if clean and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for current_dir, dirs, files in os.walk(root_dir):
        dirname = os.path.basename(current_dir)
        # 只处理包含指定关键字的目录
        if keyword not in dirname:
            continue
        # 构造 eval_best_model 子目录的路径
        eval_best_path = os.path.join(current_dir, 'eval_best_model')

        # 检查该目录是否存在，并包含 eval_log.csv 文件
        if os.path.isdir(eval_best_path):
            if 'eval_log.csv' in os.listdir(eval_best_path):
                m = pattern.match(dirname)
                if not m:
                    print(f"跳过无法解析的目录名: {dirname}")
                    continue
                meta = m.groupdict()
                prefix = meta['prefix']
                # 判断消融组 vs 对照组
                ablation_keys = ['single', 'nlg', 'k6.5', 'nobeta']
                found = [k for k in ablation_keys if k in prefix]
                ablation_type = found[0] if found else 'control'

                # 读取日志
                df = pd.read_csv(os.path.join(eval_best_path, 'eval_log.csv'))
                df['alg']      = meta['alg']
                df['attack']   = meta['attack']
                df['eps']      = float(meta['eps'])
                df['as']       = int(meta['as'])
                df['ablation'] = ablation_type
                df['seed']     = int(meta['seed'])
                df['exp_name'] = dirname
                records.append(df[['timestep', 'ep_rew_mean', 'alg', 'attack', 'eps', 'as', 'ablation', 'seed', 'exp_name']])

    if not records:
        raise RuntimeError(f"在 “{root_dir}” 下未找到任何包含 {keyword} 的实验结果目录。")

    all_df = pd.concat(records, ignore_index=True)

    # 保存所有实验记录为一个总表 CSV 文件
    csv_output_path = f'logs/{keyword}_all_experiments_eval_merged.csv'
    #csv_output_path = 'logs/MoE_rb0.5_ts500_vanilla_PC_VP_k6.5nlgPC.csv'
    all_df.to_csv(csv_output_path, mode='a', header=not os.path.exists(csv_output_path), index=False)
    print(f"Saved merged CSV: {csv_output_path}")

    # 只保留前三个 eps 值并排序（假设三个 eps）
    eps_values = sorted(all_df['eps'].unique())[:3]

    # 按算法和攻击步数分组，生成每张图片
    combos = all_df[['alg', 'as']].drop_duplicates().values.tolist()
    for alg, attack_steps in combos:
        # 限制生成 9 张图：算法*步数
        sub_df = all_df[(all_df['alg'] == alg) & (all_df['as'] == attack_steps)]
        # 创建含三个子图的画布
        fig, axes = plt.subplots(len(eps_values), 1, figsize=(8, 4 * len(eps_values)), sharex=True)
        if len(eps_values) == 1:
            axes = [axes]

        for ax, eps in zip(axes, eps_values):
            d = sub_df[sub_df['eps'] == eps]
            # 每种攻击方法与消融类型绘制一条曲线
            for (attack, ablation), grp in d.groupby(['attack', 'ablation']):
                agg = grp.groupby('timestep')['ep_rew_mean'].agg(['mean', 'sem']).reset_index()
                label = f"{attack}-{ablation}"
                ax.plot(agg['timestep'], agg['mean'], label=label)
                ax.fill_between(
                    agg['timestep'],
                    agg['mean'] - agg['sem'],
                    agg['mean'] + agg['sem'],
                    alpha=0.3
                )
            ax.set_ylabel('Mean Reward')
            ax.set_title(f"eps={eps}")
            ax.legend(fontsize='small', loc='best')

        axes[-1].set_xlabel('Timestep')
        fig.suptitle(f"{alg} | as={attack_steps}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'eval_ens2_{alg}_as{attack_steps}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved plot: {save_path}")
def collect_and_plot(root_dir: str,
                     keyword: str = 'ab_ens2',
                     clean: bool = False):
    # Pattern to extract metadata from directory names
    pattern = re.compile(
        r'^(?P<prefix>.*?)_(?P<env>TrafficEnv\d+-v\d+)_(?P<alg>\w+)_(?P<attack>[^_]+)_eps(?P<eps>[\d.]+)_as(?P<as>\d+)_(?P<seed>\d+)$'
    )

    # Prepare output directory
    output_dir = os.path.join(root_dir, 'plots')
    if clean and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for current_dir, dirs, files in os.walk(root_dir):
        dirname = os.path.basename(current_dir)
        # 只处理包含指定关键字的目录
        if keyword not in dirname:
            continue

        if 'rollout_log.csv' in files:
            # m = pattern.match(dirname)
            # if not m:
            #     print(f"跳过无法解析的目录名: {dirname}")
            #     continue
            # meta = m.groupdict()
            # prefix = meta['prefix']
            # # 判断消融组 vs 对照组
            # ablation_keys = ['single', 'nlg', 'k6.5', 'nobeta']
            # found = [k for k in ablation_keys if k in prefix]
            # ablation_type = found[0] if found else 'control'
            #
            # # 读取日志
            df = pd.read_csv(os.path.join(current_dir, 'rollout_log.csv'))
            # df['alg']      = meta['alg']
            df['alg'] = dirname.split('_')[0]
            # df['attack']   = meta['attack']
            df['attack'] = dirname.split('_')[-4]
            # df['eps']      = float(meta['eps'])
            df['eps'] = dirname.split('_')[-3]
            # df['as']       = int(meta['as'])
            df['as'] = dirname.split('_')[-2]
            # df['ablation'] = ablation_type
            df['ablation'] = 'nobeta' if 'nobeta' in dirname else 'control'
            # df['seed']     = int(meta['seed'])
            df['seed'] = dirname.split('_')[-1]
            df['exp_name'] = dirname
            records.append(df[['timestep', 'ep_rew_mean', 'alg', 'attack', 'eps', 'as', 'ablation', 'seed', 'exp_name']])


    if not records:
        raise RuntimeError(f"在 “{root_dir}” 下未找到任何包含 {keyword} 的实验结果目录。")

    all_df = pd.concat(records, ignore_index=True)

    # 保存所有实验记录为一个总表 CSV 文件
    csv_output_path = f'logs/{keyword}_all_experiments_rollout_merged.csv'
    #csv_output_path = 'logs/MoE_rb0.5_ts500_vanilla_PC_VP_k6.5nlgPC.csv'
    all_df.to_csv(csv_output_path, mode='a', header=not os.path.exists(csv_output_path), index=False)
    print(f"Saved merged CSV: {csv_output_path}")

    # 只保留前三个 eps 值并排序（假设三个 eps）
    eps_values = sorted(all_df['eps'].unique())[:3]

    # 按算法和攻击步数分组，生成每张图片
    combos = all_df[['alg', 'as']].drop_duplicates().values.tolist()
    for alg, attack_steps in combos:
        # 限制生成 9 张图：算法*步数
        sub_df = all_df[(all_df['alg'] == alg) & (all_df['as'] == attack_steps)]
        # 创建含三个子图的画布
        fig, axes = plt.subplots(len(eps_values), 1, figsize=(8, 4 * len(eps_values)), sharex=True)
        if len(eps_values) == 1:
            axes = [axes]

        for ax, eps in zip(axes, eps_values):
            d = sub_df[sub_df['eps'] == eps]
            # 每种攻击方法与消融类型绘制一条曲线
            for (attack, ablation), grp in d.groupby(['attack', 'ablation']):
                agg = grp.groupby('timestep')['ep_rew_mean'].agg(['mean', 'sem']).reset_index()
                label = f"{attack}-{ablation}"
                ax.plot(agg['timestep'], agg['mean'], label=label)
                ax.fill_between(
                    agg['timestep'],
                    agg['mean'] - agg['sem'],
                    agg['mean'] + agg['sem'],
                    alpha=0.3
                )
            ax.set_ylabel('Mean Reward')
            ax.set_title(f"eps={eps}")
            ax.legend(fontsize='small', loc='best')

        axes[-1].set_xlabel('Timestep')
        fig.suptitle(f"{alg} | as={attack_steps}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'ens2_{alg}_as{attack_steps}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved plot: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="聚合并绘制包含特定关键字的实验结果曲线 (按 eps 子图，比较攻击方法与消融)。"
    )
    parser.add_argument(
        '--root-dir', '-r',
        type=str,
        default='.',
        help='要遍历的根目录路径，默认当前目录。'
    )
    parser.add_argument(
        '--keyword', '-k',
        type=str,
        default='MoE_rb0.5',
        help='只处理目录名包含该关键字的实验结果，默认 "MoE_rb0.5"。'
    )
    parser.add_argument(
        '--clean', '-c',
        action='store_true',
        help='如果指定，先删除已有 plots/ 目录及其中内容。'
    )
    parser.add_argument(
        '--eval', '-e',
        action='store_true',
        help='如果指定，先删除已有 plots/ 目录及其中内容。'
    )
    args = parser.parse_args()
    if args.eval:
        collect_and_plot_eval(args.root_dir, args.keyword, args.clean)
    else:
        collect_and_plot(args.root_dir, args.keyword, args.clean)
