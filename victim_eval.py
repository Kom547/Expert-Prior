import argparse
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
import Environment.environment
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

from config import get_config

def make_env(env_name, attack, adv_steps, seed, max_steps):
    env = gym.make(env_name, attack=attack, adv_steps=adv_steps)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

if __name__ == '__main__':
    # get parameters from config.py
    parser = get_config()

    # parser = argparse.ArgumentParser(description='Evaluate trained agent performance')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved agent folder')
    # parser.add_argument('--algo', type=str, choices=['PPO', 'SAC', 'TD3'], required=True, help='Algorithm name')
    # parser.add_argument('--env_name', type=str, required=True, help='Gym environment ID')
    # parser.add_argument('--n_eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    # parser.add_argument('--seed', type=int, default=0, help='Random seed for evaluation env')
    # parser.add_argument('--attack', action='store_true', help='Whether environment uses attack')
    # parser.add_argument('--adv_steps', type=int, default=0, help='Number of adversarial steps')
    # parser.add_argument('--horizon', type=int, default=1000, help='Max steps per episode')
    # args = parser.parse_args()
    args = parser.parse_args()

    # Load environment
    env = make_env(args.env_name, args.attack, args.adv_steps, args.seed+50000, args.T_horizon)
    # Load agent
    if args.algo == 'PPO':
        model = PPO.load(os.path.join(args.model_path, 'best_model/best_model'), env=env)
    elif args.algo == 'SAC':
        model = SAC.load(os.path.join(args.model_path, 'best_model/best_model'), env=env)
    elif args.algo == 'TD3':
        model = TD3.load(os.path.join(args.model_path, 'lunar'), env=env)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    # # Evaluate
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.train_step, deterministic=True)
    # print(f"Evaluation over {args.train_step} episodes: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    rewards = []
    horizons_reached = []  # track if episode ran full horizon

    for ep in range(1, args.train_step + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        step_count = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        reached = (step_count >= args.T_horizon)
        rewards.append(total_reward)
        horizons_reached.append(reached)
        print(f"Episode {ep}: Reward = {total_reward:.2f}, Steps = {step_count}, Reached horizon = {reached}")

    # Summary
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    reach_rate = np.mean(horizons_reached) * 100


    env.close()
    print(f"\nEvaluation summary over {args.train_step} episodes:")
    print(f"  Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Horizon reach rate = {reach_rate:.1f}% ({sum(horizons_reached)}/{args.train_step})")
    #print(f"Evaluation over {args.train_step} episodes: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
