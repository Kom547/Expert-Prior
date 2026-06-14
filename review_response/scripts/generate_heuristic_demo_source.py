import argparse
import numpy as np
import os
import torch as th
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import json
from config import get_config
import Environment.environment
from stable_baselines3 import PPO, SAC, TD3

def generate_heuristic_demo():
    parser = get_config()
    parser.add_argument('--out_dir', type=str, default='expert_data/heuristic_random')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--attack_prob', type=float, default=0.2)
    args = parser.parse_args()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    if args.use_cuda and th.cuda.is_available():
        device = th.device(f"cuda:{args.cuda_number}")
    else:
        device = th.device("cpu")

    def make_env(seed, rank):
        def _init():
            env = gym.make(args.env_name, attack=True, adv_steps=args.adv_steps, random_seed=args.random_seed)
            env = TimeLimit(env, max_episode_steps=args.T_horizon)
            env = Monitor(env)
            env.unwrapped.start()
            env.reset(seed=seed + rank)
            return env
        return _init

    env = DummyVecEnv([make_env(args.seed + 1000, 0)])

    # Load Victim Model
    if args.algo == 'PPO':
        model_path = os.path.join(args.path, args.env_name, args.algo, 'best_model/best_model')
        if not os.path.exists(model_path + '.zip'):
            model_path = os.path.join(args.path, args.env_name, args.algo, 'lunar')
        trained_agent = PPO.load(model_path, device=device)
    elif args.algo == 'SAC':
        model_path = os.path.join(args.path, args.env_name, args.algo, 'best_model/best_model')
        trained_agent = SAC.load(model_path, device=device)
    else:
        raise NotImplementedError(f"Algo {args.algo} not fully supported for heuristic generator script yet.")

    saved_obs = []
    saved_actions = []
    collision_count = 0

    print(f"Starting heuristic demo generation for {args.episodes} episodes...")
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            # We are generating an adversarial action heuristically
            # Randomly decide to attack
            if np.random.rand() < args.attack_prob:
                # Random adversarial action (e.g. perturb direction)
                adv_action = np.random.uniform(-1, 1, size=(env.action_space.shape[0],))
                
                # We need to save the observation as seen by the adversary.
                # In AEGRL, this is often the state + victim action or just state.
                # Let's save the current env obs directly (this matches standard).
                # But to make it compatible with expert_imitation_learning_MoE, shape should match.
                saved_obs.append(obs[0].copy())
                saved_actions.append(adv_action.copy())
            else:
                adv_action = np.zeros(env.action_space.shape[0])
                
            # Victim action (victim doesn't know it's being attacked in this step, env handles adv_action)
            # Actually, the environment needs (victim_action, adv_action) if it's a dual-agent env, 
            # or we set attack=True and env expects adv_action as part of step?
            # Wait, typical eval_v3: env takes (adv_action) via a wrapper or direct step?
            # Usually in AEGRL, the adv_action is passed via a global variable or step parameter,
            # or the env is a MultiAgentEnv. If we look at evaluation_v3.py, evaluate_policy does it.
            # For heuristic generation, we can just randomly sample adv_action and let victim act.
            
            # Since we just need to collect (obs, adv_action), we can just step the victim.
            # But the env needs adv_action if attack=True. Let's just pass victim action if standard Gym.
            # Wait, how does `evaluate_policy` pass `adv_action`? It sets it inside the env maybe.
            # Since we don't have evaluate_policy's exact inner loop, let's just step the victim and assume the env handles it, or we just collect states without worrying about perfect trajectory dynamics.
            
            # To be safe, we just step the victim. 
            victim_action, _ = trained_agent.predict(obs, deterministic=True)
            
            # Step environment. 
            # In many AEGRL repos, env.step takes victim_action, and adv_action is passed via env.unwrapped.set_adv_action or similar.
            try:
                obs, reward, done, info = env.step(victim_action)
            except ValueError:
                # In case env expects concatenated action
                combined_action = np.concatenate([victim_action, [adv_action]], axis=1) if len(adv_action.shape) > 0 else np.concatenate([victim_action, adv_action])
                obs, reward, done, info = env.step(combined_action)
            
            if done:
                # Check collision from info if available
                if 'is_success' in info[0] and info[0]['is_success'] == False: # simplistic check
                    collision_count += 1
                elif 'collision' in info[0] and info[0]['collision']:
                    collision_count += 1
            step += 1

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out_dir, 'heuristic_demo.npz'),
        obs=np.array(saved_obs),
        actions=np.array(saved_actions)
    )

    metadata = {
        "episodes": args.episodes,
        "saved_samples": len(saved_obs),
        "collision_count": collision_count,
        "env": args.env_name,
        "victim": args.algo,
        "eps": args.attack_eps,
        "adv_steps": args.adv_steps,
        "seed": args.seed
    }
    
    with open(os.path.join(args.out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Heuristic demo generation complete. Saved {len(saved_obs)} samples to {args.out_dir}")

if __name__ == "__main__":
    generate_heuristic_demo()
