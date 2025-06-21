import torch
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
from Environment.environment.env8.traffic_env import Traffic_Env
import gymnasium as gym
from utils import get_trained_agent
from perturbation import FGSM_v2
from config import get_config

# 获取配置参数
parser = get_config()
args = parser.parse_args()

# 设置设备
device = torch.device(f"cuda:{args.cuda_number}" if args.use_cuda and torch.cuda.is_available() else "cpu")

# 创建环境
env = gym.make(args.env_name, attack=True, eval=True, random_seed=True)
env.unwrapped.start()

# 加载训练好的受害代理
trained_agent = get_trained_agent(args, device)

# 获取初始状态
obs, info = env.reset()
last_state = obs_as_tensor(obs[:-2], device)  # 假设 obs[:-2] 是状态部分，去掉额外信息

# 指定一个对抗动作（手动设置，确保与正常动作不同）
adv_action = torch.tensor([0.0], device=device)  # 假设动作是一维的，可根据实际动作空间调整

# 获取正常动作
with torch.no_grad():
    normal_action, _ = trained_agent.predict(last_state.cpu(), deterministic=True)
    normal_action = torch.tensor(normal_action, device=device)

# 运行 FGSM_v2 获取扰动状态
adv_state = FGSM_v2(adv_action, victim_agent=trained_agent, last_state=last_state,
                    epsilon=args.attack_eps, device=device, num_iterations=50)

# 获取攻击后的动作
with torch.no_grad():
    adv_action_after_attack, _ = trained_agent.predict(adv_state.cpu(), deterministic=True)
    adv_action_after_attack = torch.tensor(adv_action_after_attack, device=device)

# 计算距离
dist_normal_to_adv = torch.norm(normal_action - adv_action).item()
dist_attack_to_adv = torch.norm(adv_action_after_attack - adv_action).item()

# 输出结果
print(f"Normal action: {normal_action.cpu().numpy()}")
print(f"Adversarial action after attack: {adv_action_after_attack.cpu().numpy()}")
print(f"Target adversarial action: {adv_action.cpu().numpy()}")
print(f"Distance between normal and adv_action: {dist_normal_to_adv:.4f}")
print(f"Distance between attack and adv_action: {dist_attack_to_adv:.4f}")

# 判断攻击是否成功
if dist_attack_to_adv < dist_normal_to_adv:
    print("FGSM_v2 attack successful!")
else:
    print("FGSM_v2 attack failed.")

# 关闭环境
env.close()