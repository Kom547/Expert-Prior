from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch as th
import numpy as np
from config import get_config
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import Environment.environment
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback
from callback import CustomEvalCallback
import random
from buffer import PaddingRolloutBuffer, DecoupleRolloutBuffer, DecouplePaddingRolloutBuffer
from algorithm import AdversarialPPO, AdversarialDecouplePPO, AdversarialEGPPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import os


def create_model(args, env, rollout_buffer_class, device, best_model_path, tensorboard_log_dir=None, run=None):
    """
    根据 args 配置来创建 基于PPO的敌手模型。
    如果需要 wandb，返回模型时会附带 wandb 配置并同步 TensorBoard；
    否则，会把 TensorBoard 日志写到 tensorboard_log_dir。
    :param args: 系统参数
    :param env: 环境
    :param rollout_buffer_class: 回放池类型
    :param device: 模型加载设备名
    :param best_model_path: 最优模型存储路径
    :param tensorboard_log_dir: 若不使用 WandB，则传入本地 TensorBoard 日志目录
    :param run: 若使用 WandB，则传入 run 对象（同步到 WandB 的 TensorBoard）
    """
    # 先确定应该传给 SB3 的 tensorboard_log 参数
    # 如果打开了 WandB，就用 runs/{run.id}，保证 WandB 可以同步 TensorBoard
    # 否则就用本地的 tensorboard_log_dir
    if run:
        tb_dir = f"runs/{run.id}"
    else:
        tb_dir = tensorboard_log_dir

    # 根据 decouple、expert_attack 来选择具体的模型类
    if args.decouple:
        model_class = AdversarialDecouplePPO
    elif args.expert_attack:
        model_class = AdversarialEGPPO
    else:
        model_class = AdversarialPPO

    # 构造模型时，如果 tb_dir 不为 None，就把 tensorboard_log=tb_dir 传进去
    if tb_dir:
        model = model_class(
            args, best_model_path, "MlpPolicy", env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            verbose=1,
            tensorboard_log=tb_dir,
            rollout_buffer_class=rollout_buffer_class,
            device=device
        )
    else:
        # 如果 tb_dir 也是 None，就不传 tensorboard_log
        model = model_class(
            args, best_model_path, "MlpPolicy", env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            verbose=1,
            rollout_buffer_class=rollout_buffer_class,
            device=device
        )

    return model


def main():
    # get parameters from config.py
    parser = get_config()
    args = parser.parse_args()

    # log path
    eval_log_path = "./logs/adv_eval/" + os.path.join(args.adv_algo, args.env_name, args.algo, args.addition_msg)
    os.makedirs(eval_log_path, exist_ok=True)

    # —— 新增：为 TensorBoard 创建本地日志目录 ——
    tb_log_path = os.path.join(eval_log_path, "tensorboard_logs")
    os.makedirs(tb_log_path, exist_ok=True)
    # —— 以上就是为了保证不使用 WandB 时也能写日志 ——

    best_model_path = os.path.join(eval_log_path, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    eval_best_model_path = os.path.join(eval_log_path, "eval_best_model")
    os.makedirs(eval_best_model_path, exist_ok=True)

    # 设置设备
    if args.use_cuda and th.cuda.is_available():
        device = th.device(f"cuda:{args.cuda_number}")
    else:
        device = th.device("cpu")

    # 设置随机种子
    random.seed(args.cuda_seed)
    np.random.seed(args.cuda_seed)
    th.manual_seed(args.cuda_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(args.cuda_seed)
        th.cuda.manual_seed_all(args.cuda_seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    model_path = os.path.join(eval_log_path, 'model')
    os.makedirs(model_path, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=model_path)

    # 是否进行 padding
    rollout_buffer_map = {
        (True, True): DecouplePaddingRolloutBuffer,
        (True, False): PaddingRolloutBuffer,
        (False, True): DecoupleRolloutBuffer,
        (False, False): RolloutBuffer
    }
    rollout_buffer_class = rollout_buffer_map[(args.padding, args.decouple)]

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

    # 初始化 WandB 并构建敌手模型
    if not args.no_wandb:
        run_name = f"{args.attack_method}-{args.algo}-{args.addition_msg}"
        run = wandb.init(project="ExpertPriorRL", name=run_name, config=args, sync_tensorboard=True)
        # 传入 tb_log_path 和 run，使得 WandB 能同步本地 TensorBoard
        model = create_model(args, env, rollout_buffer_class, device, best_model_path, tensorboard_log_dir=tb_log_path, run=run)

        wandb_callback = WandbCallback(gradient_save_freq=500, verbose=2)
        eval_callback = CustomEvalCallback(
            eval_env,
            trained_agent=model.trained_agent,
            best_model_save_path=eval_best_model_path,
            n_eval_episodes=20,
            eval_freq=args.n_steps,
            unlimited_attack=args.unlimited_attack,
            attack_method=args.attack_method
        )
        model.learn(
            total_timesteps=args.train_step * args.n_steps,
            progress_bar=True,
            callback=[checkpoint_callback, wandb_callback, eval_callback]
        )
    else:
        # 不使用 WandB 时，只把 tb_log_path 目录传给 create_model，让 SB3 写 TensorBoard 日志
        run = None
        model = create_model(args, env, rollout_buffer_class, device, best_model_path, tensorboard_log_dir=tb_log_path, run=None)

        eval_callback = CustomEvalCallback(
            eval_env,
            trained_agent=model.trained_agent,
            best_model_save_path=eval_best_model_path,
            n_eval_episodes=20,
            eval_freq=args.n_steps,
            unlimited_attack=args.unlimited_attack,
            attack_method=args.attack_method
        )
        model.learn(
            total_timesteps=args.train_step * args.n_steps,
            progress_bar=True,
            callback=[checkpoint_callback, eval_callback]
        )

    # Save the agent
    model.save(os.path.join(eval_log_path, "lunar"))
    del model  # 删除模型示例

    env.close()


if __name__ == '__main__':
    main()
