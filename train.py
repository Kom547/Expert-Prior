import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from config import get_config
import Environment.environment
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
import numpy as np
from gymnasium.wrappers import TimeLimit
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


# get parameters from config.py
parser = get_config()
args = parser.parse_args()

#
# # Create environment
# env = gym.make(args.env_name)
#
# if args.env_name == 'TrafficEnv5-v0':
#     args.T_horizon = 80
#
# env = TimeLimit(env, max_episode_steps=args.T_horizon)
# env = Monitor(env)
# env.unwrapped.start()


def make_env(seed, rank):
    def _init():
        env = gym.make(args.env_name, attack=args.attack, adv_steps=args.adv_steps)
        env = TimeLimit(env, max_episode_steps=args.T_horizon)
        env = Monitor(env)
        env.unwrapped.start()
        env.reset(seed=seed + rank)
        return env

    return _init

num_envs = args.num_envs if hasattr(args, 'num_envs') else 1
if num_envs > 1:
    env = SubprocVecEnv([make_env(args.seed, i) for i in range(num_envs)])
    # eval_env = SubprocVecEnv([make_env(args.seed + 1000, i) for i in range(num_envs)])
else:
    env = DummyVecEnv([make_env(args.seed, 0)])
eval_env = DummyVecEnv([make_env(args.seed + 1000, 0)])

if args.no_wandb is not True:
    #init wandb
    run_name = f"{args.algo}-{args.env_name}-{args.addition_msg}"
    run = wandb.init(project="STA_AD_No_Attack", name=run_name, config=args, sync_tensorboard=True)
    wandb_callback = WandbCallback(gradient_save_freq=0, verbose=2, model_save_path=None,)

# log path
eval_log_path = "./logs/eval/" + os.path.join(args.env_name, args.algo, args.addition_msg)
eval_callback = EvalCallback(eval_env,
                             best_model_save_path=eval_log_path + "/best_model",
                             log_path=eval_log_path,  # 日志保存路径
                             eval_freq=300,  # 每150步评估一次
                             n_eval_episodes=5,  # 每次评估5个episode
                             deterministic=True)

# 设置设备
if args.use_cuda and torch.cuda.is_available():
    device = torch.device(f"cuda:{args.cuda_number}")
else:
    device = torch.device("cpu")

# Instantiate the agent
if args.no_wandb:
    if args.algo == 'PPO':
        model = PPO("MlpPolicy", env, verbose=1, device=device, n_steps=args.n_steps, n_epochs=args.n_epochs, clip_range=args.clip_range)
    elif args.algo == 'SAC':
        model = SAC("MlpPolicy", env, verbose=1, batch_size=64, device=device)
    elif args.algo == 'TD3':
        model = TD3("MlpPolicy", env, verbose=1, device=device)
else:
    if args.algo == 'PPO':
        model = PPO("MlpPolicy", env, verbose=1, device=device, n_steps=args.n_steps, n_epochs=args.n_epochs, clip_range=args.clip_range, tensorboard_log=f"runs/{run.id}")
    elif args.algo == 'SAC':
        model = SAC("MlpPolicy", env, verbose=1, batch_size=64, device=device, tensorboard_log=f"runs/{run.id}")
    elif args.algo == 'TD3':
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
        #model = TD3("MlpPolicy", env, verbose=1, device=device, tensorboard_log=f"runs/{run.id}")
        # model = TD3(
        #     "MlpPolicy",
        #     env,
        #     learning_rate=1e-4,  # 降低学习率
        #     batch_size=args.batch_size,  # 增加批量大小
        #     buffer_size=2000000,  # 增大缓冲区
        #     tau=0.001,  # 减慢目标网络更新
        #     policy_delay=3,  # 增加 actor 更新延迟
        #     target_policy_noise=0.1,  # 减小目标策略噪声
        #     target_noise_clip=0.3,  # 减小噪声剪切
        #     policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256]),optimizer_kwargs=dict(weight_decay=0.0001),),  # 增加网络容量
        #     tensorboard_log=f"runs/{run.id}",
        #     verbose=1,
        #     device=device,
        # )
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=1e-4,  # 为actor和critic分别设置学习率
            batch_size=128,  # 明确指定批大小
            buffer_size=2000000,
            tau=0.001,
            policy_delay=2,  # 增加critic训练频率
            target_policy_noise=0.1,
            target_noise_clip=0.3,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], qf=[256, 256]),
                optimizer_class=torch.optim.AdamW,  # 使用AdamW优化器
                optimizer_kwargs=dict(weight_decay=0.001),  # 调整权重衰减
            ),
            tensorboard_log=f"runs/{run.id}",
            verbose=1,
            device=device,
        )

# Train the agent and display a progress bar
if args.no_wandb:
    model.learn(total_timesteps=args.train_step*args.n_steps, progress_bar=True, callback=[eval_callback])
else:
    model.learn(total_timesteps=args.train_step*args.n_steps, progress_bar=True, callback=[eval_callback, wandb_callback])

# Save the agent
model.save(eval_log_path + "/lunar")
del model  # delete trained model to demonstrate loading
# plt fig
# 读取评估日志文件
eval_log_file = os.path.join(eval_log_path, "evaluations.npz")

# 确保文件存在
if os.path.exists(eval_log_file):
    # 加载日志文件
    data = np.load(eval_log_file)

    # 获取评估奖励
    mean_rewards = data["results"].mean(axis=1)  # 每次评估的平均奖励

    # 绘制评估奖励曲线
    plt.plot(mean_rewards)
    plt.title('Evaluation Rewards During Training')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Mean Reward')
    plt.savefig(os.path.join(eval_log_path, "mean_rewards.png"), dpi=300)
    plt.show()
else:
    print(f"Evaluation log file {eval_log_file} not found.")


env.close()

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
# model = PPO.load("ppo_lunar", env=env)



# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
# vec_env = model.get_env()
# obs = vec_env.reset()
#
# score = 0
# v = []
# v_epi = []
# sn = 0.0
# cn = 0.0
#
# for i in range(1000):
#     for j in range(30):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, rewards, dones, info = vec_env.step(action)
#         score += rewards
#         v.append(obs[24] * 15.0)
#         v_epi.append(obs[24] * 15.0)
#         xa = info[0]
#         ya = info[1]
#         if dones:
#             cn += 1
#             break
#     if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v0':
#         if xa < -50.0 and ya > 4.0 and dones is False:
#             sn += 1
#     if (i + 1) % 10 == 0:
#         print("# of episode :{}, avg score_v : {:.1f}".format(i + 1, score / args.print_interval))
#         print("######cn & sn rate:", cn / 10, sn / 10)