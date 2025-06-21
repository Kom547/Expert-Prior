# 文件名：expert_imitation_with_optuna.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import optuna
import random
from sklearn.model_selection import train_test_split
from expert_imitation_learning import Actor, ExpertDataset, calculate_test_metricsv2, ema_smooth, LossVisualize, load_expert_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = '/data/lxy/STA-Expert/expert_data/ensemblev2'  # 请替换为实际路径
save_dir = 'expert_model/ens2/optuna_model'
os.makedirs(save_dir, exist_ok=True)

state_shape = (28,)
action_dim = 2

def train_with_hyperparams(lr, batch_size, weight_decay, dropout_rate):
    OBS, ACT = load_expert_samples(file_path)
    train_obs, test_obs, train_act, test_act = train_test_split(
        OBS, ACT, test_size=0.2, random_state=42
    )

    model = Actor(state_shape, action_dim, dropout_rate=dropout_rate).to(device)

    train_dataset = ExpertDataset(train_obs, train_act)
    test_dataset = ExpertDataset(test_obs, test_act)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = 100

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            mean, std = model(x)
            var = std ** 2
            loss = 0.5 * torch.mean(torch.log(var + 1e-6) + (y - mean) ** 2 / (var + 1e-6))
            loss.backward()
            optimizer.step()

    test_loss, _, _ = calculate_test_metricsv2(model, test_loader, device)
    return test_loss

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)

    return train_with_hyperparams(lr, batch_size, weight_decay, dropout_rate)

def run_optuna():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    print("最优超参数:", study.best_params)
    return study.best_params

# 修改后的 Actor 类（带可调 Dropout）
class Actor(nn.Module):
    def __init__(self, state_shape, action_dim, dropout_rate=0.2):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(64, action_dim)
        self.std_layer = nn.Linear(64, action_dim)

    def forward(self, x):
        features = self.fc(x)
        mean = self.mean_layer(features)
        std = F.softplus(self.std_layer(features)) + 1e-6
        return mean, std

if __name__ == '__main__':
    best_params = run_optuna()

    # 用最佳参数重新训练并保存模型
    final_loss = train_with_hyperparams(**best_params)
    print("最终模型测试损失:", final_loss)
