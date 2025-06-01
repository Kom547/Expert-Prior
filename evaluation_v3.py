import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from evaluation_v4 import evaluate_policy,evaluate_policy_allinone
from evaluation_csv import evaluate_policy_csv
import numpy as np
import os
from config import get_config
import torch as th
from stable_baselines3 import SAC, PPO, TD3  # 或者您使用的其他算法

#from expert_imitation_learning import load_ensemble_models
from expert_imitation_learning_MoE import load_ensemble_models
from utils import get_attack_prob, get_trained_agent
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
import Environment.environment

# get parameters from config.py
parser = get_config()
args = parser.parse_args()

# 设置随机种子
np.random.seed(args.seed)  # 设置 NumPy 随机种子
th.manual_seed(args.seed)  # 设置 CPU 随机种子
if th.cuda.is_available():
    th.cuda.manual_seed(args.seed)  # 设置 CUDA 随机种子
    th.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 随机种子
th.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
th.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化

# 设置设备
if args.use_cuda and th.cuda.is_available():
    device = th.device(f"cuda:{args.cuda_number}")
else:
    device = th.device("cpu")

# 设置eval标志
args.eval = True

if args.attack:
    if args.adv_model_path != "":
        advmodel_path = args.adv_model_path + os.path.join('eval_best_model','best_model')
    else:
        if args.expbase_algo == '':
            args.expbase_algo = args.algo
        if args.best_model:
            if args.env_name == 'TrafficEnv8-v1' or args.env_name == 'TrafficEnv3-v5':
                advmodel_path = "./logs/adv_eval/" + os.path.join(args.adv_algo, args.env_name, args.expbase_algo, args.addition_msg,'eval_best_model','best_model')
            else:
                advmodel_path = "./logs/adv_eval/" + os.path.join(args.adv_algo, args.env_name, args.expbase_algo, args.addition_msg,'best_model')
        else:
            advmodel_path = "./logs/adv_eval/" + os.path.join(args.adv_algo, args.env_name, args.expbase_algo, args.addition_msg,'lunar')


    # 加载训练好的攻击者模型
    if args.adv_algo == 'SAC':
        model = SAC.load(advmodel_path, device=device)
    elif args.adv_algo == 'PPO':
        model = PPO.load(advmodel_path, device=device)
    else:
        model = PPO.load(advmodel_path, device=device)

    # 加载专家模型
    if args.expert_attack:
        expert_models = load_ensemble_models(args.expert_model_path, (28,), 2, device)

# 加载训练好的自动驾驶模型
#trained_agent = get_trained_agent(args, device)
# Load agent
if args.best_model:
    if args.algo == 'TD3' and args.env_name == 'TrafficEnv3-v5':
        model_path = os.path.join(args.path, args.env_name, args.algo, 'vecTD35/lunar')
    else:
        model_path = os.path.join(args.path, args.env_name, args.algo, 'best_model/best_model')
else:
    if args.algo == 'TD3' and args.env_name == 'TrafficEnv3-v5':
        model_path = os.path.join(args.path, args.env_name, args.algo, 'vecTD35/lunar')
    else:
        model_path = os.path.join(args.path, args.env_name, args.algo, 'lunar')
print('**********************************************************')

if args.algo == 'PPO':
    trained_agent = PPO.load(model_path, device=device)
elif args.algo == 'SAC':
    trained_agent = SAC.load(model_path, device=device)
elif args.algo == 'TD3':
    print(model_path)
    trained_agent = TD3.load(model_path, device=device)


def make_env(seed, rank):
    def _init():
        env = gym.make(args.env_name, attack=args.attack, adv_steps=args.adv_steps,random_seed=args.random_seed)
        env = TimeLimit(env, max_episode_steps=args.T_horizon)
        env = Monitor(env)
        env.unwrapped.start()
        env.reset(seed=seed + rank)
        return env

    return _init
eval_env = DummyVecEnv([make_env(args.seed + 50000, 0)])

# episode_rewards, episode_lengths, episode_attack_times = evaluate_policy_allinone(
#                 model,
#                 trained_agent,
#                 eval_env,
#                 expert_models = expert_models if args.expert_attack else None,
#                 n_eval_episodes=args.train_step,
#                 render=False,
#                 deterministic=True,
#                 return_episode_rewards=True,
#                 warn=True,
#                 unlimited_attack=args.unlimited_attack,
#                 attack_method=args.attack_method,
#                 save_samples=True if args.expert_recording else False,  # 新参数：控制是否保存样本
#                 save_path=f'expert_data/{args.expert_data_path}/{args.addition_msg}/'
#             )
episode_rewards, episode_lengths, episode_attack_times = evaluate_policy(
                model,
                trained_agent,
                eval_env,
                expert_models = expert_models if args.expert_attack else None,
                n_eval_episodes=args.train_step,
                render=False,
                deterministic=True,
                return_episode_rewards=True,
                warn=True,
                unlimited_attack=args.unlimited_attack,
                attack_method=args.attack_method,
                save_samples=True if args.expert_recording else False,  # 新参数：控制是否保存样本
                save_path=f'expert_data/{args.expert_data_path}/{args.addition_msg}/'
            )
# episode_rewards, episode_lengths, episode_attack_times = evaluate_policy_csv(
#                 model,
#                 trained_agent,
#                 eval_env,
#                 expert_models = expert_models if args.expert_attack else None,
#                 n_eval_episodes=args.train_step,
#                 render=False,
#                 deterministic=True,
#                 return_episode_rewards=True,
#                 warn=True,
#                 unlimited_attack=args.unlimited_attack,
#                 attack_method=args.attack_method,
#                 save_samples=True if args.expert_recording else False,  # 新参数：控制是否保存样本
#                 save_path=f'expert_data/{args.expert_data_path}/{args.env_name}_{args.algo}/'
#             )
mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
mean_attack_times, std_ep_attack_times = np.mean(episode_attack_times), np.std(episode_attack_times)
# 新增：计算置信区间
# 新增：定义计算置信区间的函数
import scipy.stats as stats  # 新增：用于计算置信区间
def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)  # 数据不足以计算置信区间
    mean = np.mean(data)
    std_err = stats.sem(data)
    dff = n - 1
    ci = stats.t.interval(confidence, dff, loc=mean, scale=std_err)
    return ci
reward_ci = calculate_confidence_interval(episode_rewards)
attack_times_ci = calculate_confidence_interval(episode_attack_times)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f} with 95% CI [{reward_ci[0]:.2f}, {reward_ci[1]:.2f}]")
print(f"mean_ep_length:{mean_ep_length:.2f} +/- {std_ep_length:.2f}")
print(f"mean_attack_times:{mean_attack_times:.2f} +/- {std_ep_attack_times:.2f} with 95% CI [{attack_times_ci[0]:.2f}, {attack_times_ci[1]:.2f}]")

if args.result_saving:
    data = {
            'env_name': [args.env_name],
            'algo': [args.algo],
            'base_algo':[args.expbase_algo],
            'expert_attack':args.expert_model_path if args.expert_attack else '',
            'method':args.expert_prior if 'Vanilla' not in args.addition_msg else 'Vanilla',
            'adv_agent':[args.addition_msg],
            'attack_eps': [args.attack_eps],
            'mean_attack_times': [mean_attack_times],
            'collision_rate': [mean_reward],
            'reward_ci_lower': [reward_ci[0]],
            'reward_ci_upper': [reward_ci[1]],
            'attack_times_ci_lower': [attack_times_ci[0]],
            'attack_times_ci_upper': [attack_times_ci[1]],
            #'attack_score':[attack_score]
        }
    df = pd.DataFrame(data)
    csv_file = f'evaluation_result/{args.result_filename}.csv'
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)