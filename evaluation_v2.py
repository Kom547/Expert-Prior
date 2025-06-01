import warnings
import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.utils import obs_as_tensor
from torch.distributed.rpc.api import method_name

from perturbation import *
import wandb
from policy import FniNet


def evaluate_policy(
    model,
    trained_agent,
    env,
    n_eval_episodes = 10,
    deterministic = True,
    render = False,
    callback = None,
    reward_threshold = None,
    return_episode_rewards = False,
    warn = True,
    unlimited_attack = False,
    attack_method = 'fgsm',
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_attack_times = []
    current_attacks = np.zeros(n_envs, dtype="int")

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(observations, model.device)
            if isinstance(trained_agent, FniNet):
                actions, std, _action = trained_agent(obs_tensor[:, :-2])
                actions = actions.detach().cpu().numpy()
            else:
                actions, _states = trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)

        actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
        obs_tensor[:, -1] = actions_tensor.squeeze(-1)

        adv_actions, states = model.predict(
            obs_tensor.cpu(),  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )

        # Get adv action mask
        if unlimited_attack:
            adv_action_mask = np.ones_like(adv_actions)
        else:
            adv_action_mask = (adv_actions[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)
            current_attacks += adv_action_mask.astype(int)

        final_actions = attack_process(obs_tensor, adv_action_mask, adv_actions, actions, attack_method, trained_agent, obs_tensor.device)

        new_observations, rewards, dones, infos = env.step(final_actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.

                            #episode_attack_times.append(round(env.envs[i].adv_steps-new_observations[i,-2]*env.envs[i].adv_steps))
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    # 记录攻击次数并重置
                    episode_attack_times.append(current_attacks[i])
                    current_attacks[i] = 0
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    #mean_attack_times = np.mean(episode_attack_times)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_attack_times
    wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})
    return mean_reward, std_reward


def attack_process(obs_tensor, adv_action_mask, clipped_adv_actions, actions, attack_method, trained_agent, device):
    if adv_action_mask.any():
        attack_idx = np.where(adv_action_mask)[0]

        selected_states = obs_tensor[attack_idx, :-2]
        print('selected_states shape:', selected_states.shape)
        selected_adv_actions = clipped_adv_actions[attack_idx, 1]
        print('selected_adv_actions shape:', selected_adv_actions)

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
        # print('clip',clipped_adv_actions,'adv_actions:', adv_actions, 'adv_final_action', action, 'actions:', actions, 'remain attack times ', obs_tensor[:, -2].cpu().numpy())
        final_action = actions.copy()
        final_action[attack_idx] = adv_action
    else:
        final_action = actions.copy()
    # Concat final_action with adv_action_mask
    output_action = np.column_stack((adv_action_mask.astype(np.float32), final_action))

    return output_action