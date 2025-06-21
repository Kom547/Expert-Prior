import warnings
import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.utils import obs_as_tensor
import os
import csv

from expert_imitation_learning_v2 import sample_from_mixture_vec
from perturbation import *
import wandb
from policy import FniNet

def evaluate_policy_csv(
    model,
    trained_agent,
    env,
    expert_models=None,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    callback=None,
    reward_threshold=None,
    return_episode_rewards=False,
    warn=True,
    unlimited_attack=False,
    attack_method='fgsm',
    save_samples=False,  # 控制是否保存样本并写入CSV
    save_path='attack_data'  # 样本保存路径
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    新增功能：收集 final_actions 并将样本以CSV格式追加写入。
    在dones[i]==1时，将该回合所有步的数据写入CSV，样本包括：done flag, obs, adv_actions, final_actions
    """
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_attack_times = []
    current_attacks = np.zeros(n_envs, dtype=int)

    episode_counts = np.zeros(n_envs, dtype=int)
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=int)

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype=int)
    observations = env.reset()
    states = None
    episode_starts = np.ones(n_envs, dtype=bool)

    # 初始化CSV与缓存
    if save_samples:
        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, 'samples.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                obs_dim = observations.shape[1]
                act_dim = 2  # assuming action dims
                writer.writerow([
                    'done',
                    *[f'obs_{i}' for i in range(obs_dim)],
                    *[f'adv_{i}' for i in range(act_dim)],
                    *[f'final_{i}' for i in range(act_dim)]
                ])
        # per-env 缓存列表
        step_buffer = [[] for _ in range(n_envs)]  # 每项: (done, obs, adv, final)

    collision_count = 0

    while (episode_counts < episode_count_targets).any():
        with th.no_grad():
            obs_tensor = obs_as_tensor(observations, model.device)
            if isinstance(trained_agent, FniNet):
                actions, std, _ = trained_agent(obs_tensor[:, :-2])
                actions = actions.detach().cpu().numpy()
            else:
                actions, _ = trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)

        actions_tensor = th.tensor(actions, device=obs_tensor.device)
        obs_tensor[:, -1] = actions_tensor.squeeze(-1)

        adv_actions, states = model.predict(
            obs_tensor.cpu(), state=states, episode_start=episode_starts, deterministic=deterministic
        )
        if expert_models is not None:
            exp_act, _, _ = sample_from_mixture_vec(obs_tensor, expert_models, device=model.device)
            adv_actions = exp_act.cpu().numpy()

        if unlimited_attack:
            adv_action_mask = np.ones_like(adv_actions)
        else:
            adv_action_mask = (adv_actions[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)
            current_attacks += adv_action_mask.astype(int)

        final_actions = attack_process(
            obs_tensor, adv_action_mask, adv_actions, actions,
            attack_method, trained_agent, obs_tensor.device
        )

        # 执行环境步，获取 dones
        new_obs, rewards, dones, infos = env.step(final_actions)

        # 缓存本步数据
        if save_samples:
            for i in range(n_envs):
                step_buffer[i].append((int(dones[i]), obs_tensor[i,:].cpu().numpy().tolist(), adv_actions[i].tolist(), final_actions[i].tolist()))

        current_rewards += rewards
        current_lengths += 1

        # 回合结束时写入CSV并清空缓存
        if save_samples:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for i in range(n_envs):
                    if dones[i] and episode_counts[i] < episode_count_targets[i]:
                        for done_flag, obs_row, adv_row, fin_row in step_buffer[i]:
                            writer.writerow([done_flag, *obs_row, *adv_row, *fin_row])
                        step_buffer[i].clear()

        # 处理回合统计
        for i in range(n_envs):
            if dones[i] and episode_counts[i] < episode_count_targets[i]:
                collision_count += 1
                if is_monitor_wrapped and "episode" in infos[i]:
                    episode_rewards.append(infos[i]["episode"]["r"])
                    episode_lengths.append(infos[i]["episode"]["l"])
                else:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                episode_attack_times.append(current_attacks[i])
                episode_counts[i] += 1
                current_attacks[i] = 0
                current_rewards[i] = 0
                current_lengths[i] = 0

        observations = new_obs
        episode_starts = dones

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, f"Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_attack_times

    wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})
    return mean_reward, std_reward


def attack_process(obs_tensor, adv_action_mask, clipped_adv_actions, actions, attack_method, trained_agent, device):
    if adv_action_mask.any():
        attack_idx = np.where(adv_action_mask)[0]
        selected_states = obs_tensor[attack_idx, :-2]
        selected_adv_actions = clipped_adv_actions[attack_idx, 1]

        if attack_method == 'fgsm':
            adv_state = FGSM_v2(selected_adv_actions, victim_agent=trained_agent, last_state=selected_states, device=device)
        elif attack_method == 'pgd':
            adv_state = PGD(selected_adv_actions, trained_agent, selected_states, device=device)
        elif attack_method == 'cw':
            adv_state = cw_attack_v2(trained_agent, selected_states, selected_adv_actions)

        if attack_method == 'direct':
            final_action = actions.copy()
            final_action[attack_idx] = (
                selected_adv_actions.detach().cpu().numpy() if th.is_tensor(selected_adv_actions) else selected_adv_actions
            )
        else:
            if isinstance(trained_agent, FniNet):
                adv_action_fromState, _, _ = trained_agent(adv_state)
                adv_action = adv_action_fromState.detach().cpu().numpy()
            else:
                adv_action_fromState, _ = trained_agent.predict(adv_state.cpu(), deterministic=True)
                adv_action = adv_action_fromState
        final_action = actions.copy()
        final_action[attack_idx] = adv_action
    else:
        final_action = actions.copy()
    return np.column_stack((adv_action_mask.astype(np.float32), final_action))
