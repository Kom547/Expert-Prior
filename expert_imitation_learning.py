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

# get parameters from config.py
parser = get_config()
# 读取参数
parser.add_argument('--samples', type=int, default=40)
args = parser.parse_args()

# GPU设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = f'expert_data/{args.env_name}_{args.algo}_{args.adv_algo}'
# 创建目录
save_dir = f'./expert_model/{args.env_name}_{args.algo}_{args.adv_algo}_{args.samples}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 定义模仿学习Actor
class Actor(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Actor, self).__init__()
        self.flatten = nn.Flatten()

        # 全连接层替代原卷积层
        self.fc = nn.Sequential(
            nn.Linear(state_shape[0], 256),  # 展平输入
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 输出层
        self.mean_layer = nn.Linear(64, action_dim)
        self.std_layer = nn.Linear(64, action_dim)

        # 初始化随机种子
        self._reset_parameters(random.randint(1, 1000))

    def _reset_parameters(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # 参数初始化
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)

        nn.init.xavier_normal_(self.mean_layer.weight)
        nn.init.constant_(self.mean_layer.bias, 0.01)
        nn.init.xavier_normal_(self.std_layer.weight)
        nn.init.constant_(self.std_layer.bias, 0.01)

    def forward(self, x):
        # x = self.flatten(x)  # 展平输入 (B, 1, 28) -> (B, 28)
        features = self.fc(x)  # 通过全连接层
        mean = self.mean_layer(features)
        std = F.softplus(self.std_layer(features)) + 1e-6  # 保证标准差非负
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


if __name__ == '__main__':
    start_time = time.time()
    # 加载处理数据
    OBS = []
    ACT = []

    files = glob.glob(file_path + '/*.npz')
    if args.samples < len(files):
        files = random.sample(files, args.samples)

    for file in files:
        data = np.load(file)
        obs = data['obs']
        act = data['adv_actions']

        for i in range(obs.shape[0]):
            OBS.append(obs[i])
            # 给动作增加一些噪声
            act[i, 0] += random.normalvariate(0, 0.1)
            act[i, 0] = np.clip(act[i, 0], -1.0, 1.0)
            act[i, 1] += random.normalvariate(0, 0.1)
            act[i, 1] = np.clip(act[i, 0], -1.0, 1.0)
            ACT.append(act[i])

    OBS = np.array(OBS, dtype=np.float32)
    ACT = np.array(ACT, dtype=np.float32)

    # 新增测试集划分代码
    from sklearn.model_selection import train_test_split

    # 划分训练集和测试集
    train_obs, test_obs, train_act, test_act = train_test_split(
        OBS, ACT, test_size=0.2, random_state=42
    )

    # 创建模型集合
    state_shape = (28,)
    action_dim = 2
    ensemble = [Actor(state_shape, action_dim).to(device) for _ in range(5)]

    # 训练参数
    epochs = 100
    EPS = 1e-6

    for idx, model in enumerate(ensemble):
        print(f'===== Training Ensemble Model {idx + 1} =====')
        # 设置随机种子
        torch.manual_seed(random.randint(1, 1000))
        np.random.seed(random.randint(1, 1000))

        # # 创建数据集和数据加载器
        # dataset = ExpertDataset(OBS, ACT)
        # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        #
        train_dataset = ExpertDataset(train_obs, train_act)
        test_dataset = ExpertDataset(test_obs, test_act)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train_loss_results = []

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

            test_accuracy = calculate_accuracy(model, test_loader)

            print(f"Progress: {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Test Distance: {test_accuracy:.4f}")
            # print(f"Progress: {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # 保存模型
        torch.save(model.state_dict(), f'{save_dir}/ensemble_{idx + 1}.pth')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"专家模型训练时间: {elapsed_time} 秒")
    with open('expert_training_time_log.txt', 'a') as file:
        file.write(save_dir+'\n')
        file.write(f"程序运行时间: {elapsed_time} 秒"+'\n')


# 加载模型集合的函数
def load_ensemble_models(save_dir, state_shape, action_dim, device='cuda'):
    models = []
    for i in range(1, 6):  # 假设保存了5个模型
        model = Actor(state_shape, action_dim).to(device)
        model.load_state_dict(torch.load(f'{save_dir}/ensemble_{i}.pth', map_location=device))
        model.eval()
        models.append(model)
    return models


# 混合高斯采样函数
def sample_from_mixture(state_np, ensemble_models, device='cuda'):
    """
    参数：
    state: 输入状态 (需为tensor)
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

    return torch.normal(mixture_mean, torch.square(mixture_var)),mixture_mean,mixture_var

# # 使用示例 ---------------------------------------------------
# # 初始化参数
# state_shape = (28,)
# action_dim = 2
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1. 加载模型
# save_path = f'./expert_model/{args.env_name}_{args.algo}_{args.adv_algo}_{args.samples}'
# ensemble_models = load_ensemble_models(save_path, state_shape, action_dim, device)

# # 2. 创建测试输入（示例）
# test_state = torch.randn(28).unsqueeze(0)  # 添加batch维度

# # 3. 进行采样
# sampled_action = sample_from_mixture(test_state, ensemble_models)
# print("Sampled action:", sampled_action.cpu().numpy())