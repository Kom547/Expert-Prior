import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd


def load_expert_samples_from_dir(directory, samples=None):
    OBS = []
    ACT = []

    files = glob.glob(os.path.join(directory, '**', '*.npz'), recursive=True)
    print(f"[{os.path.basename(directory)}] Found {len(files)} files")

    if samples and samples < len(files):
        files = random.sample(files, samples)

    for file in files:
        data = np.load(file)
        obs = data['obs']
        obs += np.random.normal(0, 0.01, obs.shape)
        act = data['adv_actions']

        for i in range(obs.shape[0]):
            OBS.append(obs[i])
            act[i, 0] += random.normalvariate(0, 0.1)
            act[i, 0] = np.clip(act[i, 0], -1.0, 1.0)
            act[i, 1] += random.normalvariate(0, 0.1)
            act[i, 1] = np.clip(act[i, 1], -1.0, 1.0)
            ACT.append(act[i])

    return np.array(OBS, dtype=np.float32), np.array(ACT, dtype=np.float32)


def load_expert_data_from_dirs(directories, samples_per_dir=None):
    all_data = {}
    for directory in directories:
        obs, act = load_expert_samples_from_dir(directory, samples=samples_per_dir)
        all_data[os.path.basename(directory)] = {'obs': obs, 'act': act}
    return all_data


def plot_tsne_comparison(obs_dict, save_path):
    all_obs = []
    labels = []

    for label, obs in obs_dict.items():
        all_obs.append(obs)
        labels.extend([label] * len(obs))

    all_obs = np.concatenate(all_obs, axis=0)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=0)
    obs_embedded = tsne.fit_transform(all_obs)

    df = pd.DataFrame(obs_embedded, columns=['dim1', 'dim2'])
    df['source'] = labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='dim1', y='dim2', hue='source', palette='tab10', s=15, alpha=0.6)
    plt.title("t-SNE comparison of observation distributions")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_kde_comparison(obs_dict, dim=0, save_path='kde_dim.png'):
    plt.figure(figsize=(8, 4))
    for label, obs in obs_dict.items():
        sns.kdeplot(obs[:, dim], label=label)
    plt.title(f"Comparison of Observation Dimension {dim}")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_action_distributions(act_dict, save_dir):
    for i in range(2):
        plt.figure(figsize=(8, 4))
        for label, act in act_dict.items():
            sns.histplot(act[:, i], kde=False, label=label, stat="density", bins=30)
        plt.title(f"Histogram of Action Dimension {i}")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'action_hist_action{i}.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 4))
        for label, act in act_dict.items():
            sns.kdeplot(act[:, i], label=label)
        plt.title(f"KDE of Action Dimension {i}")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'action_kde_action{i}.png'), bbox_inches='tight')
        plt.close()

    # Scatter plot action0 vs action1
    plt.figure(figsize=(8, 6))
    for label, act in act_dict.items():
        plt.scatter(act[:, 0], act[:, 1], label=label, alpha=0.4, s=10)
    plt.xlabel("Action 0")
    plt.ylabel("Action 1")
    plt.title("Action 0 vs Action 1")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'action_scatter.png'), bbox_inches='tight')
    plt.close()

    # t-SNE on actions
    all_act = []
    act_labels = []
    for label, act in act_dict.items():
        all_act.append(act)
        act_labels.extend([label] * len(act))

    all_act = np.concatenate(all_act, axis=0)
    tsne = TSNE(n_components=2, random_state=0)
    act_embedded = tsne.fit_transform(all_act)
    df = pd.DataFrame(act_embedded, columns=['dim1', 'dim2'])
    df['source'] = act_labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='dim1', y='dim2', hue='source', palette='tab10', s=15, alpha=0.6)
    plt.title("t-SNE of Action Distributions")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_action.png'), bbox_inches='tight')
    plt.close()


def main():
    directories = [
        '/data/lxy/STA-Expert/expert_data/TrafficEnv3-v0_PPO_PPO_FGSM_0.01',
        '/data/lxy/STA-Expert/expert_data/TrafficEnv3-v0_PPO_PPO_FGSM_0.02',
        '/data/lxy/STA-Expert/expert_data/TrafficEnv3-v0_PPO_PPO_FGSM_0.03',
        '/data/lxy/STA-Expert/expert_data/TrafficEnv3-v0_PPO_PPO_FGSM_0.04',
        '/data/lxy/STA-Expert/expert_data/TrafficEnv3-v0_PPO_PPO_FGSM_0.05',
        '/data/lxy/STA-Expert/expert_data/TrafficEnv3-v0_PPO_PPO_FGSM_0.06',
        '/data/lxy/STA-Expert/expert_data/TrafficEnv3-v0_PPO_PPO_FGSM_0.07',
        '/data/lxy/STA-Expert/expert_data/TrafficEnv3-v0_PPO_PPO_FGSM_0.08',
        '/data/lxy/STA-Expert/expert_data/TrafficEnv3-v0_PPO_PPO_FGSM_0.09',
        '/data/lxy/STA-Expert/expert_data/TrafficEnv3-v0_PPO_PPO_FGSM_0.1',
    ]

    save_dir = './plots'
    os.makedirs(save_dir, exist_ok=True)

    data_dict = load_expert_data_from_dirs(directories, samples_per_dir=50)

    obs_dict = {k: v['obs'] for k, v in data_dict.items()}
    act_dict = {k: v['act'] for k, v in data_dict.items()}

    plot_tsne_comparison(obs_dict, os.path.join(save_dir, 'tsne_obs_sources.png'))

    for dim in [0, 1, 2]:
        plot_kde_comparison(obs_dict, dim=dim, save_path=os.path.join(save_dir, f'kde_dim{dim}.png'))

    plot_action_distributions(act_dict, save_dir)


if __name__ == '__main__':
    main()