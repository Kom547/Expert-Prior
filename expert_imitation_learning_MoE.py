import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import sys
import argparse
import os
from config import get_config
import time
#from edata_ana import load_expert_samples_from_dir,load_expert_data_from_dirs
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
# get parameters from config.py
parser = get_config()
# 读取参数
parser.add_argument('--samples', type=int, default=40)
parser.add_argument('--sampling_ratio', type=float, default=1.0, help='正负样本采样比例，0<ratio<=1')
parser.add_argument('--balance', type=bool, default=True)


args = parser.parse_args()

# GPU设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.expert_data_path == '':
    file_path = f'expert_data/{args.env_name}_{args.algo}_{args.adv_algo}_{args.attack_eps}'
else:
    file_path = args.expert_data_path
# 创建目录
save_dir = f'./expert_model/{args.env_name}_{args.algo}_{args.adv_algo}_{args.attack_eps}_{args.samples}'
if args.expert_model_savepath != '':
    save_dir = args.expert_model_savepath
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 定义专家网络
class Expert(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Expert, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(64, action_dim)
        self.std_layer = nn.Linear(64, action_dim)

    def forward(self, x):
        features = self.fc(x)
        mean = self.mean_layer(features)
        std = F.softplus(self.std_layer(features)) + 1e-6
        return mean, std

# 定义模仿学习Actor（MoE结构）
class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, num_experts=3):
        super(Actor, self).__init__()
        # 创建多个专家网络
        self.experts = nn.ModuleList([Expert(state_shape, action_dim) for _ in range(num_experts)])
        # 定义门控网络
        self.gating_network = nn.Sequential(
            nn.Linear(state_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )
        # 初始化参数
        self._reset_parameters(random.randint(1, 1000))

    def _reset_parameters(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 初始化专家网络参数
        for expert in self.experts:
            for layer in expert.fc:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.constant_(layer.bias, 0.01)
            nn.init.xavier_normal_(expert.mean_layer.weight)
            nn.init.constant_(expert.mean_layer.bias, 0.01)
            nn.init.xavier_normal_(expert.std_layer.weight)
            nn.init.constant_(expert.std_layer.bias, 0.01)
        # 初始化门控网络参数
        for layer in self.gating_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)

    def forward(self, x):
        # 通过门控网络计算专家权重
        weights = F.softmax(self.gating_network(x), dim=1)  # (batch_size, num_experts)
        # 获取每个专家的输出
        experts_outputs = [expert(x) for expert in self.experts]
        means = torch.stack([m for m, _ in experts_outputs], dim=1)  # (batch_size, num_experts, action_dim)
        stds = torch.stack([s for _, s in experts_outputs], dim=1)  # (batch_size, num_experts, action_dim)
        # 调整权重维度以进行加权平均
        weights = weights.unsqueeze(2)  # (batch_size, num_experts, 1)
        # 计算加权平均的mean和std
        mean = (weights * means).sum(dim=1)  # (batch_size, action_dim)
        std = (weights * stds).sum(dim=1)  # (batch_size, action_dim)
        return mean, std

# 定义数据集
class ExpertDataset(Dataset):
    def __init__(self, obs, act):
        self.obs = obs
        self.act = act

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]

# 加载模型集合的函数
def load_ensemble_models(save_dir, state_shape, action_dim, model_cnt=6,device='cuda'):
    models = []
    for i in range(1, model_cnt):  # 假设保存了5个模型
        model = Actor(state_shape, action_dim).to(device)
        model.load_state_dict(torch.load(f'{save_dir}/ensemble_{i}.pth', map_location=device))
        model.eval()
        models.append(model)
    return models
def calculate_action_proportions(ACT):
    """
    计算ACT数据集中第一个动作维度大于0和小于0的样本比例。

    参数：
        ACT: 动作数据，形状为 (n_samples, action_dim)

    返回：
        prop_positive: 第一个动作维度大于0的样本比例
        prop_negative: 第一个动作维度小于0的样本比例
    """
    total_samples = len(ACT)
    print(total_samples)
    if total_samples == 0:
        print("警告：ACT数据集为空")
        return 0.0, 0.0

    # 计算第一个动作维度的正负样本数量
    first_action = ACT[:, 0]
    positive_count = np.sum(first_action > 0)
    negative_count = np.sum(first_action < 0)

    # 计算比例
    prop_positive = positive_count / total_samples
    prop_negative = negative_count / total_samples

    # 打印结果
    print(f"第一个动作维度 > 0 的样本比例: {prop_positive:.4f}")
    print(f"第一个动作维度 < 0 的样本比例: {prop_negative:.4f}")

    return prop_positive, prop_negative
# 混合高斯采样函数
def sample_from_mixture(state_np, ensemble_models, device='cuda'):
    """
    参数：
    state: 输入状态
    ensemble_models: 加载的模型集合
    device: 计算设备

    返回：
    sampled_action: 从混合分布中采样的动作
    """
    # 收集所有模型的输出
    means = []
    stds = []
    with torch.no_grad():
        if isinstance(state_np,torch.Tensor):
            state_tensor =state_np
        else:
            if len(state_np.shape) == 1:
                state_tensor = torch.from_numpy(state_np).unsqueeze(0).float().to(device)
            else:
                state_tensor = torch.from_numpy(state_np).float().to(device)

        state = state_tensor.to(device)
        for model in ensemble_models:
            mean, std = model(state)
            means.append(mean)
            stds.append(std)

    means_tensor = torch.stack(means).to(device)
    stds_tensor = torch.stack(stds).to(device)

    variances_tensor = stds_tensor ** 2
    # 计算混合分布的均值
    mixture_mean = torch.mean(means_tensor, dim=0)  # 沿 num_components 维度求均值

    # 计算混合分布的方差
    mixture_var = torch.mean(variances_tensor + torch.square(means_tensor), dim=0) - torch.square(mixture_mean)

    return torch.normal(mixture_mean, torch.sqrt(mixture_var)),mixture_mean,mixture_var
def sample_from_mixture_vec(state_np, ensemble_models, device='cuda'):
    """
    参数：
    state_np: 输入状态，形状为 (num_envs, obs_dim) 或 (obs_dim,) for single environment
    ensemble_models: 加载的模型集合
    device: 计算设备

    返回：
    sampled_action: 从混合分布中采样的动作，形状为 (num_envs, action_dim)
    mixture_mean: 混合分布的均值，形状为 (num_envs, action_dim)
    mixture_var: 混合分布的方差，形状为 (num_envs, action_dim)
    """
    # 处理输入
    if isinstance(state_np, torch.Tensor):
        state_tensor = state_np
    else:
        if len(state_np.shape) == 1:
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).float().to(device)
        else:
            state_tensor = torch.from_numpy(state_np).float().to(device)

    state = state_tensor.to(device)
    num_envs = state.shape[0]

    # 收集所有模型的输出
    means = []
    stds = []
    with torch.no_grad():
        for model in ensemble_models:
            mean, std = model(state)
            means.append(mean)
            stds.append(std)

    # 将列表转换为张量，形状为 (num_models, num_envs, action_dim)
    means_tensor = torch.stack(means, dim=0)
    stds_tensor = torch.stack(stds, dim=0)

    variances_tensor = stds_tensor ** 2

    # 计算混合分布的均值，形状为 (num_envs, action_dim)
    mixture_mean = torch.mean(means_tensor, dim=0)

    # 计算混合分布的方差，形状为 (num_envs, action_dim)
    mixture_var = torch.mean(variances_tensor + torch.square(means_tensor), dim=0) - torch.square(mixture_mean)

    # 从混合分布中采样动作
    sampled_action = torch.normal(mixture_mean, torch.sqrt(mixture_var))

    return sampled_action, mixture_mean, mixture_var
def ema_smooth(data, alpha=0.9):
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = alpha * last + (1 - alpha) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def calculate_test_metricsv2(model, test_loader, device):
    """
    评估模型在测试集上的性能，返回多个指标。

    参数：
        model: 训练好的模型
        test_loader: 测试集数据加载器
        device: 计算设备（'cuda' 或 'cpu'）

    返回：
        avg_loss: 平均似然损失
        avg_mse: 平均均方误差（基于采样的动作）
        avg_mae: 平均绝对误差（基于采样的动作）
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mean, std = model(x)
            var = std ** 2
            loss = 0.5 * torch.mean(torch.log(var + 1e-6) + (y - mean) ** 2 / (var + 1e-6))
            total_loss += loss.item() * x.size(0)

            sampled_actions = torch.normal(mean, std)

            # 计算均方误差（MSE）基于采样的动作
            mse = torch.mean((sampled_actions - y) ** 2)
            total_mse += mse.item() * x.size(0)

            # 计算平均绝对误差（MAE）基于采样的动作
            mae = torch.mean(torch.abs(sampled_actions - y))
            total_mae += mae.item() * x.size(0)

            total_samples += x.size(0)  # 累计样本数

    # 计算平均值
    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples

    return avg_loss, avg_mse, avg_mae
def calculate_test_metrics(model, test_loader, device):
    """
    评估模型在测试集上的性能，返回多个指标。

    参数：
        model: 训练好的模型
        test_loader: 测试集数据加载器
        device: 计算设备（'cuda' 或 'cpu'）

    返回：
        avg_loss: 平均似然损失
        avg_mse: 平均均方误差
        avg_mae: 平均绝对误差
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)  # 将数据移到指定设备
            mean, std = model(x)  # 模型预测均值和标准差
            var = std ** 2  # 计算方差

            # 计算似然损失
            loss = 0.5 * torch.mean(torch.log(var + 1e-6) + (y - mean) ** 2 / (var + 1e-6))
            total_loss += loss.item() * x.size(0)

            # 计算均方误差（MSE）
            mse = torch.mean((mean - y) ** 2)
            total_mse += mse.item() * x.size(0)

            # 计算平均绝对误差（MAE）
            mae = torch.mean(torch.abs(mean - y))
            total_mae += mae.item() * x.size(0)

            total_samples += x.size(0)  # 累计样本数

    # 计算平均值
    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples

    return avg_loss, avg_mse, avg_mae
def calculate_accuracy(model, test_loader):
    model.eval()
    total_distance = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mean, std = model(x)
            # 从高斯分布采样
            sampled_actions = torch.normal(mean, std)
            # 计算欧式距离并取平均
            distances = torch.norm(sampled_actions - y, dim=1)
            total_distance += distances.sum().item()
    return total_distance / len(test_loader.dataset)

def LossVisualize(save_dir, all_train_losses,all_test_losses):
    # 画图
    plt.figure(figsize=(12, 6))

    # 子图1：训练Loss
    for idx, train_losses in enumerate(all_train_losses):
        plt.plot(train_losses, label=f'Model {idx + 1} Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss for All Ensemble Models')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/ensemble_train_loss.png')
    plt.close()

    # 子图2：测试Loss（即测试集平均距离）
    plt.figure(figsize=(12, 6))
    for idx, test_losses in enumerate(all_test_losses):
        plt.plot(test_losses, label=f'Model {idx + 1} Test Distance')
    plt.xlabel('Epoch')
    plt.ylabel('Test Distance')
    plt.title('Test Distance for All Ensemble Models')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/ensemble_test_distance.png')
    plt.close()


def load_expert_samples(filename=''):
    OBS = []
    ACT = []

    # 获取所有子文件夹路径（只包含直接子文件夹）
    subfolders = [f.path for f in os.scandir(filename) if f.is_dir()]

    for folder in subfolders:
        # 查找当前文件夹及其子文件夹中的所有npz文件
        files = glob.glob(os.path.join(folder, '**', '*.npz'), recursive=True)

        if not files:
            continue  # 跳过没有npz文件的文件夹

        # 计算需要抽取的文件数量（至少1个）
        sample_size = max(1, int(len(files) * 0.1))
        selected_files = random.sample(files, sample_size)

        for file in selected_files:
            data = np.load(file)
            obs = data['obs']
            #obs += np.random.normal(0, 0.01, obs.shape)
            act = data['adv_actions']
            #print(obs.shape,  act.shape)
            for i in range(obs.shape[0]):
                OBS.append(obs[i])
                # 给动作增加一些噪声
                act[i, 0] += random.normalvariate(0, 0.1)
                act[i, 0] = np.clip(act[i, 0], -1.0, 1.0)
                act[i, 1] += random.normalvariate(0, 0.1)
                act[i, 1] = np.clip(act[i, 1], -1.0, 1.0)
                ACT.append(act[i])

    OBS = np.array(OBS, dtype=np.float32)
    ACT = np.array(ACT, dtype=np.float32)

    return OBS, ACT

def visualize_tsne(data_dir,
                   use_concat=True,
                   balance_samples=False,
                   save_path=None,
                   perplexity=30,
                   n_components=2,
                   random_state=42):
    """
    对 data_dir 下所有 .npz 文件做 t-SNE，可视化正负样本。

    参数：
        data_dir (str): 顶层目录，函数会遍历其所有子目录并查找 .npz 文件
        use_concat (bool): True 使用 obs 与 adv_actions 拼接；False 只用 obs。
        balance_samples (bool): True 对负样本进行随机下采样，正负样本数量相同。
        save_path (str or None): 如果不为 None，则保存图片到该路径；否则直接 plt.show()
        perplexity (int): TSNE perplexity 参数
        n_components (int): TSNE 输出维度（一般2）
        random_state (int): 随机种子
    """
    features = []
    labels = []

    # 收集所有 .npz 文件
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    all_files = []
    for folder in subfolders:
        all_files += glob.glob(os.path.join(folder, '**', '*.npz'), recursive=True)

    # 构造特征和标签
    for file in all_files:
        data = np.load(file)
        obs = data['obs']
        acts = data['adv_actions'] if use_concat else None
        n = obs.shape[0]
        for i in range(n):
            if use_concat:
                feat = np.concatenate([obs[i], acts[i]])
            else:
                feat = obs[i]
            features.append(feat)
            labels.append(1 if i == n-1 else 0)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)

    # 平衡正负样本
    if balance_samples:
        pos_idx = np.where(y == 1)[0].tolist()
        neg_idx = np.where(y == 0)[0].tolist()
        if len(neg_idx) > len(pos_idx):
            neg_idx = random.sample(neg_idx, len(pos_idx))
        keep_idx = pos_idx + neg_idx
        X = X[keep_idx]
        y = y[keep_idx]
    print('Sample Complete.Start TSNE.')
    # t-SNE 降维
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                random_state=random_state)
    X_tsne = tsne.fit_transform(X)

    # 绘图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0],
                          X_tsne[:, 1],
                          c=y,
                          cmap='bwr',
                          alpha=0.6,
                          s=10)
    # 自定义图例
    neg_patch = mpatches.Patch(color='blue', label='negative')
    pos_patch = mpatches.Patch(color='red', label='positive')
    plt.legend(handles=[neg_patch, pos_patch])
    plt.title('t-SNE: positive vs. negative')
    plt.xlabel('TSNE dim1')
    plt.ylabel('TSNE dim2')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"已保存 t-SNE 图到: {save_path}")
    else:
        plt.show()

def visualize_tsne_v2(data_dir,
                   use_concat=True,
                   balance_samples=False,
                   sampling_ratio=1.0,
                   save_dir=None,
                   perplexity=30,
                   n_components=2,
                   random_state=42):
    """
    对 data_dir 下根据算法子目录分别做 t-SNE 可视化。

    参数：
        data_dir (str): 顶层目录，函数会遍历其所有子目录并查找包含算法名称(SAC, TD3, PPO)的子目录
        use_concat (bool): True 使用 obs 与 adv_actions 拼接；False 只用 obs。
        balance_samples (bool): True 对负样本进行随机下采样，正负样本数量相同。
        sampling_ratio (float): 样本采样比例 (0<ratio<=1)，用于降密度
        save_dir (str or None): 如果不为 None，则保存各算法的图片到该目录；否则直接 plt.show()
        perplexity (int): TSNE perplexity 参数
        n_components (int): TSNE 输出维度（一般2）
        random_state (int): 随机种子
    """
    # 定义优先级算法列表
    alg_order = ['SAC', 'TD3', 'PPO']
    # 收集每个算法对应的文件列表
    alg_files = {alg: [] for alg in alg_order}
    for entry in os.scandir(data_dir):
        if not entry.is_dir():
            continue
        name = entry.name.upper()
        for alg in alg_order:
            if alg in name:
                # PPO 只有在 SAC/TD3 均不出现时才归为 PPO
                if alg == 'PPO' and any(a in name for a in ['SAC', 'TD3']):
                    continue
                alg_files[alg].append(entry.path)
                break
    # 逐算法处理
    for alg in alg_order:
        paths = alg_files[alg]
        if not paths:
            continue
        features, labels = [], []
        # 遍历该算法下所有 npz
        for folder in paths:
            for file in glob.glob(os.path.join(folder, '**', '*.npz'), recursive=True):
                data = np.load(file)
                obs = data['obs']
                acts = data['adv_actions'] if use_concat else None
                n = obs.shape[0]
                for i in range(n):
                    feat = np.concatenate([obs[i], acts[i]]) if use_concat else obs[i]
                    features.append(feat)
                    labels.append(1 if i == n-1 else 0)
        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32)
        # 平衡正负样本
        if balance_samples:
            pos_idx = np.where(y == 1)[0].tolist()
            neg_idx = np.where(y == 0)[0].tolist()
            if len(neg_idx) > len(pos_idx):
                neg_idx = random.sample(neg_idx, len(pos_idx))
            keep = pos_idx + neg_idx
            X = X[keep]
            y = y[keep]
        # 按比例采样
        if sampling_ratio < 1.0:
            idx = list(range(len(y)))
            sample_n = max(1, int(len(idx) * sampling_ratio))
            idx = random.sample(idx, sample_n)
            X = X[idx]
            y = y[idx]
        # t-SNE 降维
        tsne = TSNE(n_components=n_components,
                    perplexity=perplexity,
                    random_state=random_state)
        X_tsne = tsne.fit_transform(X)
        # 绘图
        plt.figure(figsize=(8, 6))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='bwr', alpha=0.6, s=10)
        neg_patch = mpatches.Patch(color='blue', label='negative')
        pos_patch = mpatches.Patch(color='red', label='positive')
        plt.legend(handles=[neg_patch, pos_patch])
        plt.title(f't-SNE：{alg}')
        plt.xlabel('TSNE dim1')
        plt.ylabel('TSNE dim2')
        plt.tight_layout()
        # 保存或展示
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_out = os.path.join(save_dir, f'tsne_{alg}.png')
            plt.savefig(file_out, dpi=300)
            plt.close()
            print(f"已保存 {alg} t-SNE 图到: {file_out}")
        else:
            plt.show()
def train():
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start_time = time.time()
    # 加载处理数据
    OBS, ACT = load_expert_samples(file_path)
    print(OBS.shape, ACT.shape)


    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    train_obs, test_obs, train_act, test_act = train_test_split(
        OBS, ACT, test_size=0.2, random_state=42
    )

    # 检查 ACT[:, 0] == 0 的样本数量
    zero_count = np.sum(ACT[:, 0] == 0)
    print(f"ACT[:, 0] == 0 的样本数量: {zero_count}")
    if args.balance:
        # 找到正样本和负样本的索引
        pos_indices = np.where(train_act[:, 0] > 0)[0]
        neg_indices = np.where(train_act[:, 0] < 0)[0]
        P = len(pos_indices)  # 正样本数量
        Q = len(neg_indices)  # 负样本数量

        if Q > P:
            # 对负样本进行下采样
            selected_neg_indices = np.random.choice(neg_indices, P, replace=False)
            selected_indices = np.concatenate([pos_indices, selected_neg_indices])
        else:
            # 如果负样本不足，保留所有负样本和正样本
            selected_indices = np.concatenate([pos_indices, neg_indices])
    else:
        # 不平衡样本时，使用所有训练数据
        selected_indices = np.arange(len(train_act))


    # 新增：按比例下采样
    if args.sampling_ratio < 1.0:
        total_sel = len(selected_indices)
        num_to_keep = max(1, int(total_sel * args.sampling_ratio))
        selected_indices = np.random.choice(selected_indices, num_to_keep, replace=False)

    np.random.shuffle(selected_indices)  # 打乱索引以混合样本

    # 创建采样后的训练数据集
    train_obs_selected = train_obs[selected_indices]
    train_act_selected = train_act[selected_indices]

    # 打印采样前后的样本数量
    print(f"原始训练样本数: {len(train_obs)}")
    print(f"采样后训练样本数: {len(train_obs_selected)}")

    # 验证采样后的动作比例
    print("采样后训练集动作比例:")
    calculate_action_proportions(train_act_selected)

    # 创建模型集合
    state_shape = (28,)
    action_dim = 2
    ensemble = [Actor(state_shape, action_dim).to(device) for _ in range(5)]

    # 训练参数
    epochs = 500
    EPS = 1e-6
    weight_decay = 1e-4
    batch_size = 128
    lr = 3e-4
    base_seed = 1000
    # 早停参数
    patience = 20  # 如果验证集MSE连续20个epoch没有改善，则停止训练
    min_delta = 1e-4  # 认为MSE改善的最小变化量

    all_train_losses = []
    all_test_losses = []

    for idx, model in enumerate(ensemble):
        print(f'===== Training Ensemble Model {idx + 1} =====')
        # 早停变量
        best_mse = float('inf')
        best_epoch = 0
        epochs_no_improve = 0
        best_model_path = f'{save_dir}/ensemble_{idx + 1}_best.pth'


        seed = base_seed + idx
        model._reset_parameters(seed)

        # 创建数据集和数据加载器
        train_dataset = ExpertDataset(train_obs_selected, train_act_selected)
        #train_dataset = ExpertDataset(train_obs, train_act)
        test_dataset = ExpertDataset(test_obs, test_act)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss_results = []
        test_mse_results = []  # 改为记录MSE

        for epoch in range(epochs):
            epoch_loss = []
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                mean, std = model(x)
                var = std ** 2
                loss = 0.5 * torch.mean(torch.log(var + EPS) + (y - mean) ** 2 / (var + EPS))

                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())

            avg_loss = np.mean(epoch_loss)
            train_loss_results.append(avg_loss)

            # 调用新的评估函数
            test_loss, test_mse, test_mae = calculate_test_metricsv2(model, test_loader, device)
            test_mse_results.append(test_mse)  # 记录MSE作为主要测试指标
            print(f"Progress: {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}, Test MSE: {test_mse}, Test MAE: {test_mae}")
            # 早停逻辑
            if test_mse < best_mse - min_delta:
                best_mse = test_mse
                best_epoch = epoch
                epochs_no_improve = 0
                # 保存最佳模型
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1} for model {idx + 1}. Best MSE: {best_mse:.4f} at epoch {best_epoch + 1}")
                # 加载最佳模型作为最终模型
                #model.load_state_dict(torch.load(best_model_path, map_location=device))
                break
        # 保存模型
        print(f'Saving model {idx} at {save_dir}/ensemble_{idx + 1}.pth')
        torch.save(model.state_dict(), f'{save_dir}/ensemble_{idx + 1}.pth')
        all_train_losses.append(train_loss_results)
        all_test_losses.append(ema_smooth(test_mse_results, alpha=0.9))

    end_time = time.time()
    LossVisualize(save_dir, all_train_losses, all_test_losses)
    elapsed_time = end_time - start_time
    print(f"专家模型训练时间: {elapsed_time} 秒")
    with open('expert_training_time_log.txt', 'a') as file:
        file.write(save_dir + '\n')
        file.write(f'epochs: {epochs}' + '\n')
        file.write(f'EPS: {EPS}' + '\n')
        file.write(f'weight_decay: {weight_decay}' + '\n')
        file.write(f'batch_size: {batch_size}' + '\n')
        file.write(f'lr: {lr}' + '\n')
        file.write(f"Last Training Loss: {all_train_losses[-1]}" + '\n')
        file.write(f"Last Test MSE: {all_test_losses[-1]}" + '\n')
        file.write(f"程序运行时间: {elapsed_time} 秒" + '\n')


if __name__ == '__main__':
    train()
    # 加载处理数据
    # OBS, ACT = load_expert_samples(file_path)
    # calculate_action_proportions(ACT)
    #visualize_tsne(file_path, True,save_path=file_path +'/tsne_plot_concat.png')
    #visualize_tsne(file_path, False, save_path=file_path + '/tsne_plot.png')
    #visualize_tsne_v2(file_path,True,False,save_dir=file_path)

