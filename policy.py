"""
Policy network for FNI
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Module(nn.Module):
    def __call__(self, *args, **kwargs):
        args = [x if isinstance(x, torch.Tensor) else x for x in args]
        kwargs = {k: v if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return super().__call__(*args, **kwargs)

    def save(self, f, prefix='', keep_vars=False):
        state_dict = self.state_dict(prefix=prefix, keep_vars=keep_vars)
        torch.save(state_dict, f)

    def load(self, f, map_location='', strict=True):
        state_dict = torch.load(f, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

class FniNet(Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=256, min_log_std=-10.0, max_log_std=10.0):
        super(FniNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)

        self.mu_head = nn.Linear(hidden_sizes, action_dim)
        self.log_std_head = nn.Linear(hidden_sizes, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(torch.clamp(self.log_std_head(x), self.min_log_std, self.max_log_std)).sqrt()

        action = mu + std * torch.randn_like(mu)
        action = torch.clamp(action, -1.0, 1.0)

        return mu, std, action

class SumoNet(nn.Module):
    def __init__(self, obs_space, action_space, hidden_layers):
        """
        初始化前馈神经网络

        :param obs_space: 状态空间的维度（一维向量的长度）
        :param action_space: 动作空间的维度（这里固定为2）
        :param hidden_layers: 隐藏层的数量和每层神经元数量（如：[64, 128, 64]）
        """
        super(SumoNet, self).__init__()

        # 特征提取网络
        layers = [nn.Linear(obs_space, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())

        self.feature_extractor = nn.Sequential(*layers)

        # 策略网络 (用于生成动作)
        self.policy_net = nn.Linear(hidden_layers[-1], action_space)

        # 值函数网络 (用于生成状态价值)
        self.value_net = nn.Linear(hidden_layers[-1], 1)

        # 动作标准差的参数 (用于构建分布)
        self.log_std = nn.Parameter(torch.zeros(action_space))

    def forward(self, obs, deterministic=False):
        """
        前向传播函数

        :param obs: 观察输入 (batch_size, obs_dim)
        :param deterministic: 是否使用确定性动作
        :return: 动作值, 状态值, 动作对数概率
        """
        # 提取特征
        features = self.feature_extractor(obs)

        # 策略网络生成动作均值
        mean_actions = self.policy_net(features)

        # 动作分布 (高斯分布)
        std = torch.exp(self.log_std)  # 将 log 标准差转为标准差
        distribution = Normal(mean_actions, std)

        # 采样或确定性动作
        if deterministic:
            actions = mean_actions
        else:
            actions = distribution.rsample()  # 采样动作 (带梯度的采样)

        # 动作对数概率
        log_prob = distribution.log_prob(actions).sum(dim=-1)

        # 值函数网络生成状态价值
        values = self.value_net(features)

        # 调整动作输出到 [-1, 1] 范围
        actions = torch.tanh(actions)

        return actions, values, log_prob

class DarrlNet(Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=256, min_log_std=-5.0, max_log_std=2.0):
        super(DarrlNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)

        self.mu_head = nn.Linear(hidden_sizes, action_dim)
        self.log_std_head = nn.Linear(hidden_sizes, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(torch.clamp(self.log_std_head(x), self.min_log_std, self.max_log_std)).sqrt()

        action = mu + std * torch.randn_like(mu)
        action = torch.clamp(action, -1.0, 1.0)

        return mu, std, action


