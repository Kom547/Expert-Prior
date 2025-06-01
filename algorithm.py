from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
import torch as th
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from gymnasium import spaces
from buffer import PaddingRolloutBuffer, DecouplePaddingRolloutBuffer
from stable_baselines3.common.utils import explained_variance
from perturbation_v2 import *
from stable_baselines3 import PPO, SAC, TD3
import os
from utils import get_attack_prob
#from expert_imitation_learning import Actor,load_ensemble_models,sample_from_mixture
from expert_imitation_learning_MoE import Actor,load_ensemble_models,sample_from_mixture
import torch.distributions as dist
from torch.nn import functional as F
import csv

# 设置全局打印选项，指定小数点后精度为 4 位
th.set_printoptions(precision=2)
class OnPolicyAdversarialAlgorithm_EG(OnPolicyAlgorithm):
    """
    Rewrite the collect_rollouts class of OnPolicyAlgorithm to support adversarial training.
    """
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration = 1
        self.last_epi_reward = 0
        # 初始化CSV文件，写入表头
        if not os.path.exists(self.rollout_log_path):
            with open(self.rollout_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestep', 'ep_rew_mean'])
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Get adv action
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if self.fni_flag:
                    actions, std, _action = self.trained_agent(obs_tensor[:, :-2])
                    actions = actions.detach().cpu().numpy()
                else:
                    actions, _states = self.trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)
                actions_tensor = th.tensor(actions, device=self.device)
                obs_tensor[:, -1] = actions_tensor.squeeze(-1)
                adv_actions, adv_values, adv_log_probs = self.policy(obs_tensor)
                if self.decouple:
                    # print('adv actions ', adv_actions)
                    distribution = self.policy.get_distribution(obs_tensor)
                    adv_log_probs = distribution.distribution.log_prob(adv_actions)

            adv_actions = adv_actions.cpu().numpy()
            # Rescale and perform action
            clipped_adv_actions = adv_actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_adv_actions = self.policy.unscale_action(clipped_adv_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_adv_actions = np.clip(adv_actions, self.action_space.low, self.action_space.high)

            # Get adv action mask
            if self.unlimited_attack:
                adv_action_mask = np.ones_like(clipped_adv_actions)
            else:
                adv_action_mask = (clipped_adv_actions[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)

            # Generate perturbation into observations to get adv_obs
            final_actions = self.attack_process(obs_tensor, adv_action_mask, clipped_adv_actions, actions)

            new_obs, rewards, dones, infos = env.step(final_actions)

            # Get next origin action according next state and insert into next_obs
            with th.no_grad():
                next_obs_tensor = obs_as_tensor(new_obs, self.device)
                if self.fni_flag:
                    actions, std, _action = self.trained_agent(next_obs_tensor[:, :-2])
                    actions = actions.cpu().numpy()
                else:
                    actions, _states = self.trained_agent.predict(next_obs_tensor[:, :-2].cpu(), deterministic=True)
                actions_tensor = th.tensor(actions, device=self.device)

                # Update next_obs_tensor
                next_obs_tensor[:, -1] = actions_tensor.squeeze(-1)
                new_obs = next_obs_tensor.detach().cpu().numpy()

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)


            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                adv_actions = adv_actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            # TODO: whether to add trajectory padding method
            # check rollout_buffer type
            # is_padding_buffer = isinstance(rollout_buffer, (PaddingRolloutBuffer, DecouplePaddingRolloutBuffer))
            # if is_padding_buffer:
            #     # 遍历每个环境的 info，处理每条轨迹
            #     for env_idx, info in enumerate(infos):
            #         if info.get('flag', False):  # 如果这个环境提前终止（攻击成功）
            #             rollout_buffer.log_collisions(env_idx=env_idx)  # 支持按环境记录碰撞
            #             rollout_buffer.return_pos(env_idx=env_idx)  # 返回当前环境的n_steps，用于后续对齐处理

            # print('last obs is ', self._last_obs)
            # print('obs tensor is ', obs_tensor.cpu().numpy())
            rollout_buffer.add(
                obs_tensor.cpu().numpy(),  # type: ignore[arg-type]
                adv_actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                adv_values,
                adv_log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            # print('last episode start is ', self._last_episode_starts)

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def attack_process(self, obs_tensor, adv_action_mask, clipped_adv_actions, actions):
        if adv_action_mask.any():
            attack_idx = np.where(adv_action_mask)[0]

            selected_states = obs_tensor[attack_idx, :-2]
            print('selected_states shape:', selected_states.shape)
            selected_adv_actions = clipped_adv_actions[attack_idx, 1]
            print('selected_adv_actions shape:', selected_adv_actions)

            if self.attack_method == 'fgsm':
                adv_state = FGSM_v2(selected_adv_actions, victim_agent=self.trained_agent,
                                    last_state=selected_states, epsilon=self.attack_eps, device=self.device)
            elif self.attack_method == 'pgd':
                adv_state = PGD(selected_adv_actions, self.trained_agent, selected_states, epsilon=self.attack_eps,
                                device=self.device)

            if self.attack_method == 'direct':
                final_action = actions.copy()
                final_action[attack_idx] = selected_adv_actions.detach().cpu().numpy() if th.is_tensor(
                    selected_adv_actions) else selected_adv_actions
            else:
                if self.fni_flag:
                    adv_action_fromState, _, _ = self.trained_agent(adv_state)
                    adv_action = adv_action_fromState.detach().cpu().numpy()
                else:
                    adv_action_fromState, _ = self.trained_agent.predict(adv_state.cpu(), deterministic=True)
                    adv_action = adv_action_fromState
            # print('clip',clipped_adv_actions,'adv_actions:', adv_actions, 'adv_final_action', action, 'actions:', actions, 'remain attack times ', obs_tensor[:, -2].cpu().numpy())
            final_action = actions.copy()
            final_action[attack_idx] = adv_action
        else:
            final_action = actions.copy()
        # Concat final_action with adv_action_mask
        output_action = np.column_stack((adv_action_mask.astype(np.float32), final_action))

        return output_action

    def _dump_logs(self, iteration):
        super()._dump_logs(iteration)

        self.iteration = iteration
        self.last_epi_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
        # 保存到CSV
        with open(self.rollout_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.num_timesteps, self.last_epi_reward])
        if self.last_epi_reward >= self.max_epi_reward:
            self.save(self.best_model_path)
            self.max_epi_reward = self.last_epi_reward

    from typing import TypeVar
    SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")
    from stable_baselines3.common.type_aliases import MaybeCallback
    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            assert self.ep_info_buffer is not None
            self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

class OnPolicyAdversarialAlgorithm_v2(OnPolicyAlgorithm):
    """
    Rewrite the collect_rollouts class of OnPolicyAlgorithm to support adversarial training.
    """
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化CSV文件，写入表头
        if not os.path.exists(self.rollout_log_path):
            with open(self.rollout_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestep', 'ep_rew_mean'])
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Get adv action
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if self.fni_flag:
                    actions, std, _action = self.trained_agent(obs_tensor[:, :-2])
                    actions = actions.detach().cpu().numpy()
                else:
                    actions, _states = self.trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)
                actions_tensor = th.tensor(actions, device=self.device)
                obs_tensor[:, -1] = actions_tensor.squeeze(-1)
                adv_actions, adv_values, adv_log_probs = self.policy(obs_tensor)
                if self.decouple:
                    # print('adv actions ', adv_actions)
                    distribution = self.policy.get_distribution(obs_tensor)
                    adv_log_probs = distribution.distribution.log_prob(adv_actions)

            adv_actions = adv_actions.cpu().numpy()
            # Rescale and perform action
            clipped_adv_actions = adv_actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_adv_actions = self.policy.unscale_action(clipped_adv_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_adv_actions = np.clip(adv_actions, self.action_space.low, self.action_space.high)

            # Get adv action mask
            if self.unlimited_attack:
                adv_action_mask = np.ones_like(clipped_adv_actions)
            else:
                adv_action_mask = (clipped_adv_actions[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)

            # Generate perturbation into observations to get adv_obs
            final_actions = self.attack_process(obs_tensor, adv_action_mask, clipped_adv_actions, actions)

            new_obs, rewards, dones, infos = env.step(final_actions)

            # Get next origin action according next state and insert into next_obs
            with th.no_grad():
                next_obs_tensor = obs_as_tensor(new_obs, self.device)
                if self.fni_flag:
                    actions, std, _action = self.trained_agent(next_obs_tensor[:, :-2])
                    actions = actions.cpu().numpy()
                else:
                    actions, _states = self.trained_agent.predict(next_obs_tensor[:, :-2].cpu(), deterministic=True)
                actions_tensor = th.tensor(actions, device=self.device)

                # Update next_obs_tensor
                next_obs_tensor[:, -1] = actions_tensor.squeeze(-1)
                new_obs = next_obs_tensor.detach().cpu().numpy()

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)


            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                adv_actions = adv_actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            # TODO: whether to add trajectory padding method
            # check rollout_buffer type
            # is_padding_buffer = isinstance(rollout_buffer, (PaddingRolloutBuffer, DecouplePaddingRolloutBuffer))
            # if is_padding_buffer:
            #     # 遍历每个环境的 info，处理每条轨迹
            #     for env_idx, info in enumerate(infos):
            #         if info.get('flag', False):  # 如果这个环境提前终止（攻击成功）
            #             rollout_buffer.log_collisions(env_idx=env_idx)  # 支持按环境记录碰撞
            #             rollout_buffer.return_pos(env_idx=env_idx)  # 返回当前环境的n_steps，用于后续对齐处理

            # print('last obs is ', self._last_obs)
            # print('obs tensor is ', obs_tensor.cpu().numpy())
            rollout_buffer.add(
                obs_tensor.cpu().numpy(),  # type: ignore[arg-type]
                adv_actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                adv_values,
                adv_log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            # print('last episode start is ', self._last_episode_starts)

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def attack_process(self, obs_tensor, adv_action_mask, clipped_adv_actions, actions):
        if adv_action_mask.any():
            attack_idx = np.where(adv_action_mask)[0]

            selected_states = obs_tensor[attack_idx, :-2]
            print('selected_states shape:', selected_states.shape)
            selected_adv_actions = clipped_adv_actions[attack_idx, 1]
            print('selected_adv_actions shape:', selected_adv_actions)

            if self.attack_method == 'fgsm':
                adv_state = FGSM_v2(selected_adv_actions, victim_agent=self.trained_agent,
                                    last_state=selected_states, epsilon=self.attack_eps, device=self.device)
            elif self.attack_method == 'pgd':
                adv_state = PGD(selected_adv_actions, self.trained_agent, selected_states, epsilon=self.attack_eps,
                                device=self.device)

            if self.attack_method == 'direct':
                final_action = actions.copy()
                final_action[attack_idx] = selected_adv_actions.detach().cpu().numpy() if th.is_tensor(
                    selected_adv_actions) else selected_adv_actions
            else:
                if self.fni_flag:
                    adv_action_fromState, _, _ = self.trained_agent(adv_state)
                    adv_action = adv_action_fromState.detach().cpu().numpy()
                else:
                    adv_action_fromState, _ = self.trained_agent.predict(adv_state.cpu(), deterministic=True)
                    adv_action = adv_action_fromState
            # print('clip',clipped_adv_actions,'adv_actions:', adv_actions, 'adv_final_action', action, 'actions:', actions, 'remain attack times ', obs_tensor[:, -2].cpu().numpy())
            final_action = actions.copy()
            final_action[attack_idx] = adv_action
        else:
            final_action = actions.copy()
        # Concat final_action with adv_action_mask
        output_action = np.column_stack((adv_action_mask.astype(np.float32), final_action))

        return output_action

    def _dump_logs(self, iteration):
        super()._dump_logs(iteration)
        last_epi_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
        # 保存到CSV
        with open(self.rollout_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.num_timesteps, last_epi_reward])
        if safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]) >= self.max_epi_reward:
            self.save(self.best_model_path)
            self.max_epi_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])

    from typing import TypeVar
    SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")
    from stable_baselines3.common.type_aliases import MaybeCallback
    def learn(
            self: SelfOnPolicyAlgorithm,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "OnPolicyAlgorithm",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            assert self.ep_info_buffer is not None
            self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

class OnPolicyAdversarialAlgorithm(OnPolicyAlgorithm):
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, total_steps, *args, **kwargs):
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            # print('current steps ', env.env_method('get_current_steps'), 'last episode start ', self._last_episode_starts)
            # print('current steps ', env.env_method('get_current_steps'), 'n_steps ', n_steps, 'n_rollout_steps ', n_rollout_steps, 'last obs is ', self._last_obs)
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                if self.fni_flag:
                    actions, std, _action = self.trained_agent(obs_tensor[:, :-2])
                    actions = actions.detach().cpu().numpy()
                else:
                    actions, _states = self.trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)

                # 假设 actions 是一个 NumPy 数组
                actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
                # obs_tensor[:, -1] = (actions_tensor + 1) / 2
                obs_tensor[:, -1] = actions_tensor
                # print('current steps ', env.env_method('get_current_steps'), 'n_steps ', n_steps, 'n_rollout_steps ',
                #       n_rollout_steps, 'actions is', actions_tensor, 'obs tensor is ', obs_tensor)

                adv_actions, adv_values, adv_log_probs = self.policy(obs_tensor)
                if isinstance(self, AdversarialDecouplePPO):
                    # print('adv actions ', adv_actions)
                    distribution = self.policy.get_distribution(obs_tensor)
                    adv_log_probs = distribution.distribution.log_prob(adv_actions)
                # print('adv log probs', adv_log_probs)

            adv_actions = adv_actions.cpu().numpy()

            if kwargs.get('use_act'):
                # # epsilon-greedy choose action
                # print('current steps ', env.env_method('get_current_steps'), 'n_steps ', n_steps, 'n_rollout_steps ', n_rollout_steps)

                # aggregation_prob
                # top_k = 6
                # k = min(top_k, len(act_list))
                # smallest_rates = np.partition(act_list, k-1)[:k]
                # print('smallest_rates', smallest_rates)
                # individual_probs = 1 / (1 + np.exp(alpha * (smallest_rates - beta)))
                # attack_prob = np.mean(individual_probs)
                # attack_prob = np.clip(attack_prob, 0, 1)

                # calculate attack prob
                act_list = env.env_method('get_act')[0]
                attack_prob = get_attack_prob(act_list)
                # calculate epsilon
                # epsilon_min = 0
                # epsilon_max = 1
                # decay_rate = 10
                # epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * self.num_timesteps / total_steps)
                # real_prob = np.clip(epsilon * attack_prob + (1 - epsilon) * adv_actions[:, 0][0], 0, 1)
                # print('epsilon', epsilon, 'attack prob ', attack_prob, 'adv actions ', adv_actions[:, 0][0], 'real prob ', real_prob)
                # # print('attack prob is ', attack_prob)
                # # print('epsilon prob is ', epsilon * attack_prob)
                # if np.random.rand() < real_prob:
                #     adv_actions[:, 0] = np.abs(adv_actions[:, 0])
                # else:
                #     adv_actions[:, 0] = -np.abs(adv_actions[:, 0])

                epsilon_min = 0
                epsilon_max = 1
                decay_rate = 10
                epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * self.num_timesteps / total_steps)
                if np.random.rand() < epsilon:
                    if np.random.rand() < attack_prob:
                        adv_actions[:, 0] = np.abs(adv_actions[:, 0])
                    # else:
                    #     adv_actions[:, 0] = -np.abs(adv_actions[:, 0])

            act_list = env.env_method('get_act')[0]
            attack_prob = get_attack_prob(act_list)

            # Rescale and perform action
            clipped_adv_actions = adv_actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_adv_actions = self.policy.unscale_action(clipped_adv_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_adv_actions = np.clip(adv_actions, self.action_space.low, self.action_space.high)

            adv_action_mask = (clipped_adv_actions[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)
            if adv_action_mask or kwargs.get('unlimited_attack', False):
                if kwargs.get('attack_method') == 'fgsm':
                    adv_state = FGSM_v2(clipped_adv_actions[:, 1], victim_agent=self.trained_agent,
                                        last_state=obs_tensor[:, :-2], device=self.device)
                elif kwargs.get('attack_method') == 'pgd':
                    adv_state = PGD(clipped_adv_actions[:, 1], self.trained_agent, obs_tensor[:, :-2],
                                    device=self.device)

                if kwargs.get('attack_method') == 'direct':
                    action = clipped_adv_actions[:, 1]
                else:
                    with th.no_grad():
                        if self.fni_flag:
                            adv_action_fromState, _, _ = self.trained_agent(adv_state)
                            action = adv_action_fromState.detach().cpu().numpy()
                        else:
                            adv_action_fromState, _states = self.trained_agent.predict(adv_state.cpu(), deterministic=True)
                            action = adv_action_fromState
                # print('clip',clipped_adv_actions,'adv_actions:', adv_actions, 'adv_final_action', action, 'actions:', actions, 'remain attack times ', obs_tensor[:, -2].cpu().numpy())
            else:
                action = actions
            action = np.column_stack((action, adv_action_mask))
            new_obs, rewards, dones, infos = env.step(action)

            # print('current steps ', env.env_method('get_current_steps'), 'n_steps ', n_steps, 'n_rollout_steps ',
            #       n_rollout_steps, 'old new obs tensor is ', new_obs)

            with th.no_grad():
                next_obs_tensor = obs_as_tensor(new_obs, self.device)
                if self.fni_flag:
                    actions, std, _action = self.trained_agent(next_obs_tensor[:, :-2])
                    actions = actions.cpu().numpy()
                else:
                    actions, _states = self.trained_agent.predict(next_obs_tensor[:, :-2].cpu(), deterministic=True)

                # 假设 actions 是一个 NumPy 数组
                actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
                # next_obs_tensor[:, -1] = (actions_tensor + 1) / 2
                next_obs_tensor[:, -1] = actions_tensor
                new_obs = next_obs_tensor.detach().cpu().numpy()

            # print('current steps ', env.env_method('get_current_steps'), 'n_steps ', n_steps, 'n_rollout_steps ',
            #       n_rollout_steps, 'action is', actions_tensor, 'new obs tensor is ', new_obs)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                adv_actions = adv_actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            if isinstance(rollout_buffer, (PaddingRolloutBuffer, DecouplePaddingRolloutBuffer)) and infos[0]['flag']:
                rollout_buffer.log_collisions()

            rollout_buffer.add(
                obs_tensor.cpu().numpy(),  # type: ignore[arg-type]
                adv_actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                adv_values,
                adv_log_probs,
                [attack_prob],
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            if isinstance(rollout_buffer, (PaddingRolloutBuffer, DecouplePaddingRolloutBuffer)) and infos[0]['flag']:
                n_steps = rollout_buffer.return_pos()
            # print('last episode start is ', self._last_episode_starts)

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def learn(self, total_timesteps, callback=None, log_interval=1,
              tb_log_name='PPO', reset_num_timesteps=True, progress_bar=False, *args, **kwargs):

        # 在这里使用额外的参数，如 attack_method 和 unlimited_attack
        # attack_method = kwargs.get('attack_method', None)
        # unlimited_attack = kwargs.get('unlimited_attack', False)

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps, total_steps=total_timesteps, *args, **kwargs)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

    def _dump_logs(self, iteration):
        super()._dump_logs(iteration)
        if safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]) >= self.max_epi_reward:
            self.save(self.best_model_path)
            self.max_epi_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])


class AdversarialPPO(OnPolicyAdversarialAlgorithm_v2, PPO):
    def __init__(self, custom_args, best_model_path, *args, **kwargs):
        self.rollout_log_path = "./logs/adv_eval/" + os.path.join(custom_args.adv_algo, custom_args.env_name, custom_args.algo, custom_args.addition_msg) + "/rollout_log.csv"

        super().__init__(*args, **kwargs)
        self.best_model = custom_args.best_model
        self.algo = custom_args.algo
        self.path = custom_args.path
        self.env_name = custom_args.env_name
        self.fni_model_path = custom_args.fni_model_path
        self.unlimited_attack = custom_args.unlimited_attack
        self.attack_method = custom_args.attack_method
        self.decouple = custom_args.decouple
        self.attack_eps = custom_args.attack_eps

        # Instantiate the agent
        # if self.best_model:
        #     model_path = os.path.join(self.path, self.env_name, self.algo, 'best_model/best_model')
        # else:
        #     model_path = os.path.join(self.path, self.env_name, self.algo, 'lunar')
        if self.best_model:
            if custom_args.algo == 'TD3' and custom_args.env_name == 'TrafficEnv3-v5':
                model_path = os.path.join(custom_args.path, custom_args.env_name, custom_args.algo, 'vecTD35/lunar')
            else:
                model_path = os.path.join(custom_args.path, custom_args.env_name, custom_args.algo, 'best_model/best_model')
        else:
            if custom_args.algo == 'TD3' and custom_args.env_name == 'TrafficEnv3-v5':
                model_path = os.path.join(custom_args.path, custom_args.env_name, custom_args.algo, 'vecTD35/lunar')
            else:
                model_path = os.path.join(custom_args.path, custom_args.env_name, custom_args.algo, 'lunar')
        if self.algo == 'PPO':
            self.trained_agent = PPO.load(model_path, device=self.device)
        elif self.algo == 'SAC':
            self.trained_agent = SAC.load(model_path, device=self.device)
        elif self.algo == 'TD3':
            self.trained_agent = TD3.load(model_path, device=self.device)
        elif self.algo == 'RA_PPO':
            self.trained_agent = PPO.load(model_path, device=self.device)
        elif self.algo in ('FNI', 'DARRL'):
            if self.env_name == 'TrafficEnv5-v0':
                state_dim = 29
                action_dim = 2
            else:
                state_dim = 26
                action_dim = 1
            if self.algo == 'FNI':
                # 创建一个新的 Actor 实例
                self.trained_agent = FniNet(state_dim, action_dim)
                self.trained_agent.load_state_dict(
                    th.load(os.path.join(self.path, self.env_name, self.algo, self.fni_model_path) + '.pth',
                            weights_only=True))
                self.trained_agent.to(self.device)
                self.trained_agent.eval()
            elif self.algo == 'DARRL':
                self.trained_agent = FniNet(state_dim, action_dim)
                self.trained_agent.load_state_dict(
                    th.load(os.path.join(self.path, self.env_name, self.algo, self.fni_model_path) + '.pth',
                            weights_only=True))
                self.trained_agent.to(self.device)
                self.trained_agent.eval()
        self.max_epi_reward = 0
        self.best_model_path = best_model_path

        # Get customized parameters
        self.fni_flag = True if self.algo in ('FNI', 'DARRL') else False



    def learn(self, total_timesteps, callback=None, log_interval=1,
              tb_log_name='PPO', reset_num_timesteps=True, progress_bar=False):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _setup_model(self):
        super()._setup_model()
        # Use the custom rollout buffer if provided, otherwise, use the default one
        if self.rollout_buffer_class is not None:
            self.rollout_buffer = self.rollout_buffer_class(
                self.n_steps,
                self.observation_space,
                self.action_space,
                self.device,
                self.gamma,
                self.gae_lambda,
                self.n_envs
            )

class AdversarialEGPPO(OnPolicyAdversarialAlgorithm_EG, PPO):
    def __init__(self, custom_args, best_model_path, *args, **kwargs):
        self.rollout_log_path = "./logs/adv_eval/" + os.path.join(custom_args.adv_algo, custom_args.env_name, custom_args.algo, custom_args.addition_msg) + "/rollout_log.csv"
        super().__init__(*args, **kwargs)
        self.best_model = custom_args.best_model
        self.algo = custom_args.algo
        self.path = custom_args.path
        self.env_name = custom_args.env_name
        self.fni_model_path = custom_args.fni_model_path
        self.unlimited_attack = custom_args.unlimited_attack
        self.attack_method = custom_args.attack_method
        self.decouple = custom_args.decouple
        self.attack_eps = custom_args.attack_eps
        self.aargs = custom_args
        # Instantiate the agent
        # if self.best_model:
        #     model_path = os.path.join(self.path, self.env_name, self.algo, 'best_model/best_model')
        # else:
        #     model_path = os.path.join(self.path, self.env_name, self.algo, 'lunar')
        if self.best_model:
            if custom_args.algo == 'TD3' and custom_args.env_name == 'TrafficEnv3-v5':
                model_path = os.path.join(custom_args.path, custom_args.env_name, custom_args.algo, 'vecTD35/lunar')
            else:
                model_path = os.path.join(custom_args.path, custom_args.env_name, custom_args.algo, 'best_model/best_model')
        else:
            if custom_args.algo == 'TD3' and custom_args.env_name == 'TrafficEnv3-v5':
                model_path = os.path.join(custom_args.path, custom_args.env_name, custom_args.algo, 'vecTD35/lunar')
            else:
                model_path = os.path.join(custom_args.path, custom_args.env_name, custom_args.algo, 'lunar')
        if self.algo == 'PPO':
            self.trained_agent = PPO.load(model_path, device=self.device)
        elif self.algo == 'SAC':
            self.trained_agent = SAC.load(model_path, device=self.device)
        elif self.algo == 'TD3':
            self.trained_agent = TD3.load(model_path, device=self.device)
        elif self.algo == 'RA_PPO':
            self.trained_agent = PPO.load(model_path, device=self.device)
        elif self.algo in ('FNI', 'DARRL'):
            if self.env_name == 'TrafficEnv5-v0':
                state_dim = 29
                action_dim = 2
            else:
                state_dim = 26
                action_dim = 1
            if self.algo == 'FNI':
                # 创建一个新的 Actor 实例
                self.trained_agent = FniNet(state_dim, action_dim)
                self.trained_agent.load_state_dict(
                    th.load(os.path.join(self.path, self.env_name, self.algo, self.fni_model_path) + '.pth',
                            weights_only=True))
                self.trained_agent.to(self.device)
                self.trained_agent.eval()
            elif self.algo == 'DARRL':
                self.trained_agent = FniNet(state_dim, action_dim)
                self.trained_agent.load_state_dict(
                    th.load(os.path.join(self.path, self.env_name, self.algo, self.fni_model_path) + '.pth',
                            weights_only=True))
                self.trained_agent.to(self.device)
                self.trained_agent.eval()
        self.max_epi_reward = 0
        self.best_model_path = best_model_path

        # Get customized parameters
        self.fni_flag = True if self.algo in ('FNI', 'DARRL') else False

        # for expert prior
        self.no_lambda_grad = custom_args.no_lambda_grad
        if custom_args.expert_attack:
            if custom_args.expert_prior == 'ValuePenalty':
                self.alpha = 0.01
            elif custom_args.expert_prior == 'PolicyConstrained':
                self.alpha = 0.01
                # 在对数空间内优化
                self.lambda_raw = th.nn.Parameter(
                    th.tensor(np.log(self.alpha), dtype=th.float32, device=self.device)
                )
                self.lambda_optimizer = th.optim.Adam([self.lambda_raw], lr=1e-4)
            self.epsilon = 0.5
            self.beta = 1
            #self.k = 0.5
            self.k = custom_args.expert_k
            self.last_stage_eprewards = []
            self.wo_beta = custom_args.wo_beta
            # 专家模型目录
            if custom_args.expert_model_path == '':
                expert_model_path = '/data/lxy/STA-Expert/expert_model/ensemblev2_2'
            else:
                expert_model_path = custom_args.expert_model_path
            self.expert_cnt = custom_args.expert_cnt
            self.expert_models = load_ensemble_models(expert_model_path, (28,), 2,self.expert_cnt, self.device)



    def learn(self, total_timesteps, callback=None, log_interval=1,
              tb_log_name='PPO', reset_num_timesteps=True, progress_bar=False):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _setup_model(self):
        super()._setup_model()
        # Use the custom rollout buffer if provided, otherwise, use the default one
        if self.rollout_buffer_class is not None:
            self.rollout_buffer = self.rollout_buffer_class(
                self.n_steps,
                self.observation_space,
                self.action_space,
                self.device,
                self.gamma,
                self.gae_lambda,
                self.n_envs
            )
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # expert intervine
        if self.iteration % 5 == 0:
            if self.last_stage_eprewards == []:
                self.beta = 1
            else:

                last_stage_mer = np.mean(self.last_stage_eprewards)
                self.beta = (1-last_stage_mer)**self.k
            self.last_stage_eprewards = []
        else:
            self.last_stage_eprewards.append(self.last_epi_reward)

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # expert_attack logs
        exp_kls = []
        lambda_vals, lambda_losses = [], []
        exp_miu, exp_v = [], []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            exp_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                actions = rollout_data.actions

                #获取专家动作分布用于计算kl散度
                obs = rollout_data.observations
                if self.aargs.expert_attack:
                    _ , expert_means, expert_vars = sample_from_mixture(obs, self.expert_models,
                                                             device=self.device)
                    exp_miu.append(expert_means.mean().item())
                    exp_v.append(expert_vars.mean().item())
                    # 计算专家分布和当前动作分布的kl散度
                    current_dist = self.policy.get_distribution(obs).distribution
                    expert_dist = th.distributions.normal.Normal(expert_means, torch.sqrt(expert_vars))
                    exp_kl = dist.kl_divergence(current_dist, expert_dist)
                    exp_kl = torch.mean(exp_kl, dim=1)
                    exp_kls.append(exp_kl.mean().item())

                    # print(exp_kl.shape)

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                #print(policy_loss_1.shape, policy_loss_2.shape)
                if self.aargs.expert_attack and self.aargs.expert_prior == 'ValuePenalty':
                    if self.wo_beta:
                        policy_loss = (-th.min(policy_loss_1, policy_loss_2) + self.alpha * exp_kl).mean()
                    else:
                        policy_loss = (-th.min(policy_loss_1, policy_loss_2) + self.beta * self.alpha * exp_kl).mean()
                    #policy_loss = (-th.min(policy_loss_1, policy_loss_2) +  self.alpha * exp_kl).mean()
                elif self.aargs.expert_attack and self.aargs.expert_prior == 'PolicyConstrained':
                    if self.no_lambda_grad:
                        policy_loss = (-th.min(policy_loss_1, policy_loss_2) + self.beta*self.alpha * (
                                    exp_kl - self.epsilon)).mean()
                    else:
                        # 计算当前lambda值（确保非负）
                        lambda_val = th.exp(self.lambda_raw)
                        policy_loss = (-th.min(policy_loss_1, policy_loss_2) + lambda_val.detach() * (exp_kl-self.epsilon)).mean()
                else:
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                #print(values.shape)
                if self.aargs.expert_attack and self.aargs.expert_prior == 'ValuePenalty':
                    if self.wo_beta:
                        value_loss = F.mse_loss(rollout_data.returns - self.alpha * exp_kl, values_pred)
                    else:
                        value_loss = F.mse_loss(rollout_data.returns - self.beta * self.alpha * exp_kl, values_pred )
                    #value_loss = F.mse_loss(rollout_data.returns -  self.alpha * exp_kl, values_pred)
                else:
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                if self.aargs.expert_attack and self.aargs.expert_prior == 'PolicyConstrained' and not self.no_lambda_grad:
                    lambda_loss = lambda_val * (self.epsilon - exp_kl.mean().detach())
                    lambda_vals.append(lambda_val.item())
                    lambda_losses.append(lambda_loss.item())
                    # 对偶优化lambda
                    self.lambda_optimizer.zero_grad()
                    lambda_loss.backward()
                    self.lambda_optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if self.aargs.expert_attack:
            self.logger.record("train/exp_std", torch.sqrt(expert_vars).mean().item())
            self.logger.record("train/exp_mean", expert_means.mean().item())
            self.logger.record("train/exp_kl", np.mean(exp_kls))
            self.logger.record("train/beta", self.beta)
            if self.aargs.expert_prior == 'PolicyConstrained':
                self.logger.record("train/lambda_val", np.mean(lambda_vals))
                self.logger.record("train/lambda_loss", np.mean(lambda_losses))

class AdversarialDecouplePPO(AdversarialPPO):
    def learn(self, total_timesteps, callback=None, log_interval=1,
              tb_log_name='PPO', reset_num_timesteps=True, progress_bar=False, *args, **kwargs):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            *args, **kwargs
        )

    def _setup_model(self):
        super()._setup_model()
        # Use the custom rollout buffer if provided, otherwise, use the default one
        if self.rollout_buffer_class is not None:
            self.rollout_buffer = self.rollout_buffer_class(
                self.n_steps,
                self.observation_space,
                self.action_space,
                self.device,
                self.gamma,
                self.gae_lambda,
                self.n_envs
            )

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        divide loss to swi_loss and lur_loss, similar to the setting of 'Attacking Deep Reinforcement Learning with  Decoupled Adversarial Policy'
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        prob_losses = []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 分离 a1 和 a2 的动作
                actions_a1 = actions[:, 0]

                # print('values', values, 'log prob', log_prob, 'entropy', entropy)
                # print('actions are ', actions)
                # print('actions', actions)
                distribution = self.policy.get_distribution(rollout_data.observations)
                log_prob_joint = distribution.distribution.log_prob(actions)
                # print('log prob joint ', log_prob_joint)
                # print('old log prob ', rollout_data.old_log_prob)
                # print('log_prob', log_prob)
                # 分离 log_prob
                log_prob_a1 = log_prob_joint[:, 0]  # [batch_size]
                log_prob_a2 = log_prob_joint[:, 1]  # [batch_size]
                # ratio between old and new policy for a1 and a2
                ratio_a1 = th.exp(log_prob_a1 - rollout_data.old_log_prob[:, 0])
                ratio_a2 = th.exp(log_prob_a2 - rollout_data.old_log_prob[:, 1])

                # clipped surrogate loss for a1 and a2
                policy_loss_1_a1 = advantages * ratio_a1
                policy_loss_2_a1 = advantages * th.clamp(ratio_a1, 1 - clip_range, 1 + clip_range)
                policy_loss_a1 = -th.min(policy_loss_1_a1, policy_loss_2_a1).mean()

                policy_loss_1_a2 = advantages * ratio_a2
                policy_loss_2_a2 = advantages * th.clamp(ratio_a2, 1 - clip_range, 1 + clip_range)
                policy_loss_a2 = -th.min(policy_loss_1_a2, policy_loss_2_a2).mean()

                # Logging
                pg_losses.append([policy_loss_a1.item(), policy_loss_a2.item()])
                clip_fraction_a1 = th.mean((th.abs(ratio_a1 - 1) > clip_range).float()).item()
                clip_fraction_a2 = th.mean((th.abs(ratio_a2 - 1) > clip_range).float()).item()
                clip_fractions.append((clip_fraction_a1 + clip_fraction_a2) / 2)

                # 引入掩码：仅在 a1 >= 0 时计算 a2 的损失
                mask = (actions_a1 >= 0).float().unsqueeze(-1)  # [batch_size, 1]
                policy_loss_a2 = (policy_loss_a2 * mask).mean()

                # 总策略损失
                policy_loss = policy_loss_a1 + policy_loss_a2

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss)

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob_joint - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


