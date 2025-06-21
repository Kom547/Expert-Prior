import warnings
import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.utils import obs_as_tensor
import os

from expert_imitation_learning_v2 import sample_from_mixture_vec
from perturbation import *
import wandb
from policy import FniNet

def evaluate_policy(
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
    save_samples=False,
    save_path='attack_data'
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    :param expert_models: Optional expert models for sampling actions.
    :param model: The RL agent to evaluate.
    :param trained_agent: The trained agent to evaluate against.
    :param env: The gym environment or VecEnv environment.
    :param n_eval_episodes: Number of episodes to evaluate the agent.
    :param deterministic: Whether to use deterministic or stochastic actions.
    :param render: Whether to render the environment.
    :param callback: Callback function called after each step.
    :param reward_threshold: Minimum expected reward per episode.
    :param return_episode_rewards: If True, returns per-episode rewards and lengths.
    :param warn: If True, warns about lack of Monitor wrapper.
    :param unlimited_attack: If True, allows unlimited attacks.
    :param attack_method: Method used for attacks (e.g., 'fgsm').
    :param save_samples: If True, saves all observation and adversarial action pairs.
    :param save_path: Directory path to save samples as a single .npz file.
    :return: Mean reward per episode, std of reward per episode.
        If ``return_episode_rewards`` is True, returns ([float], [int], [int]) for
        rewards, lengths, and attack times per episode.
    """
    is_monitor_wrapped = False
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
    current_attacks = np.zeros(n_envs, dtype="int")

    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    if save_samples:
        all_obs = []
        all_adv_actions = []
        os.makedirs(save_path, exist_ok=True)

    collision_count = 0

    while (episode_counts < episode_count_targets).any():
        with th.no_grad():
            obs_tensor = obs_as_tensor(observations, model.device)
            if isinstance(trained_agent, FniNet):
                actions, std, _action = trained_agent(obs_tensor[:, :-2])
                actions = actions.detach().cpu().numpy()
            else:
                actions, _states = trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)

        actions_tensor = th.tensor(actions, device=obs_tensor.device)
        obs_tensor[:, -1] = actions_tensor.squeeze(-1)

        adv_actions, states = model.predict(
            obs_tensor.cpu(),
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        if expert_models is not None:
            state_tensor = obs_tensor
            expert_action, _, _ = sample_from_mixture_vec(state_tensor, expert_models, device=model.device)
            expert_action = expert_action.cpu().numpy()
            adv_actions = expert_action

        if unlimited_attack:
            adv_action_mask = np.ones_like(adv_actions)
        else:
            adv_action_mask = (adv_actions[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)
            current_attacks += adv_action_mask.astype(int)

        if save_samples:
            for i in range(n_envs):
                all_obs.append(obs_tensor[i, :].cpu().numpy())
                all_adv_actions.append(adv_actions[i, :])

        final_actions = attack_process(obs_tensor, adv_action_mask, adv_actions, actions, attack_method, trained_agent, obs_tensor.device)

        new_observations, rewards, dones, infos = env.step(final_actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    collision_count += 1
                    if is_monitor_wrapped:
                        if "episode" in info.keys():
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    episode_attack_times.append(current_attacks[i])
                    current_attacks[i] = 0
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, f"Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_attack_times

    if save_samples:
        all_obs = np.array(all_obs, dtype=np.float32)
        all_adv_actions = np.array(all_adv_actions, dtype=np.float32)
        file_name = "all_samples.npz"
        file_path = os.path.join(save_path, file_name)
        print("Saving all samples at:", file_path)
        np.savez_compressed(
            file_path,
            obs=all_obs,
            adv_actions=all_adv_actions
        )

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
            final_action[attack_idx] = selected_adv_actions.detach().cpu().numpy() if th.is_tensor(
                selected_adv_actions) else selected_adv_actions
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
    output_action = np.column_stack((adv_action_mask.astype(np.float32), final_action))
    return output_action