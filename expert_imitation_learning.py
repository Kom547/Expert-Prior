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
from DatasetTNSEAnalyse import load_expert_samples_from_dir,load_expert_data_from_dirs

# get parameters from config.py
parser = get_config()
# 读取参数
parser.add_argument('--samples', type=int, default=40)
args = parser.parse_args()




# 定义模仿学习Actor
class Actor(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(Actor, self).__init__()
        self.flatten = nn.Flatten()

        # 全连接层替代原卷积层
        # self.fc = nn.Sequential(
        #     nn.Linear(state_shape[0], 256),  # 展平输入
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU()
        # )
        self.fc = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
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

    return torch.normal(mixture_mean, torch.sqrt(mixture_var)),mixture_mean,mixture_var

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
            obs += np.random.normal(0, 0.01, obs.shape)
            act = data['adv_actions']

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

    # 新增测试集划分代码
    from sklearn.model_selection import train_test_split

    # 划分训练集和测试集
    train_obs, test_obs, train_act, test_act = train_test_split(
        OBS, ACT, test_size=0.2, random_state=42
    )

    # 检查 ACT[:, 0] == 0 的样本数量
    zero_count = np.sum(ACT[:, 0] == 0)
    print(f"ACT[:, 0] == 0 的样本数量: {zero_count}")
    # 对训练数据进行采样
    # 找到第一个动作维度大于0的样本索引
    pos_indices = np.where(train_act[:, 0] > 0)[0]
    # 找到第一个动作维度小于0的样本索引
    neg_indices = np.where(train_act[:, 0] < 0)[0]
    P = len(pos_indices)  # 正样本数量
    Q = len(neg_indices)  # 负样本数量
    num_neg_to_select = 3 * P  # 需要选择的负样本数量

    # 根据负样本数量选择采样方式
    if Q >= num_neg_to_select:
        # 如果负样本足够，不放回采样
        selected_neg_indices = np.random.choice(neg_indices, num_neg_to_select, replace=False)
    else:
        # 如果负样本不足，放回采样
        selected_neg_indices = np.random.choice(neg_indices, num_neg_to_select, replace=True)

    # 合并正样本和选择的负样本索引
    selected_indices = np.concatenate([pos_indices, selected_neg_indices])

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

    #origin
    epochs =200
    EPS = 1e-6
    lr = 3e-4
    batch_size = 32
    #weight_decay = 1e-5
    base_seed = 1000

    # # 训练参数
    # epochs = 500
    # EPS = 1e-6
    # weight_decay = 1e-5
    # #weight_decay = 7.753706163922767e-06
    # batch_size = 32
    # #batch_size = 64
    # lr = 3e-4
    # #lr = 0.0008080703408898859

    all_train_losses = []
    all_test_losses = []

    for idx, model in enumerate(ensemble):
        print(f'===== Training Ensemble Model {idx + 1} =====')
        # 设置随机种子
        # torch.manual_seed(random.randint(1, 1000))
        # np.random.seed(random.randint(1, 1000))
        seed = base_seed + idx
        model._reset_parameters(seed)
        # # 创建数据集和数据加载器
        # dataset = ExpertDataset(OBS, ACT)
        # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        #
        train_dataset = ExpertDataset(train_obs_selected, train_act_selected)
        #train_dataset = ExpertDataset(train_obs, train_act)
        test_dataset = ExpertDataset(test_obs, test_act)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_loss_results = []
        test_accuracy_results = []

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
            #test_mse_results.append(test_mse)  # 记录MSE作为主要测试指标

            #test_accuracy = calculate_accuracy(model, test_loader)
            test_accuracy_results.append(test_loss)
            print(f"Progress: {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Test Distance: {test_loss:.4f}")
            # print(f"Progress: {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # 保存模型
        torch.save(model.state_dict(), f'{save_dir}/ensemble_{idx + 1}.pth')
        all_train_losses.append(train_loss_results)
        all_test_losses.append(ema_smooth(test_accuracy_results, alpha=0.9))

    end_time = time.time()
    LossVisualize(save_dir, all_train_losses, all_test_losses)
    elapsed_time = end_time - start_time
    print(f"专家模型训练时间: {elapsed_time} 秒")
    with open('expert_training_time_log.txt', 'a') as file:
        file.write(save_dir + '\n')
        file.write(f"程序运行时间: {elapsed_time} 秒" + '\n')


if __name__ == '__main__':
    train()
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