import pandas as pd
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch as th
import numpy as np
from config import get_config
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor, safe_mean,explained_variance, get_schedule_fn
from torch.nn import functional as F
from stable_baselines3.common.type_aliases import MaybeCallback
import Environment.environment
from typing import Optional
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib
# 设置 matplotlib 的后端为 Agg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from FGSM import *
from stable_baselines3.common.buffers import RolloutBuffer
import wandb
from wandb.integration.sb3 import WandbCallback
import random

from expert_imitation_learning import Actor,load_ensemble_models,sample_from_mixture
import torch.distributions as dist


class CustomRolloutBuffer(RolloutBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, gamma=0.99, gae_lambda=1, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma,
                                                  gae_lambda, n_envs)
        self.max_steps = 30  # Assuming buffer_size as max_steps
        self.current_episode_length = 0
        self.flag = False

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        # print('episode_starts is ', self.episode_starts[self.pos-1])
        if self.episode_starts[self.pos-1][0]:
            self.current_episode_length = 0
        else:
            self.current_episode_length += 1
        # Check if the episode has ended
        if self.flag:
            # print('*****************************padding***************************************')
            # If episode length < max_steps, repeat the trajectory
            if self.current_episode_length < self.max_steps:
                N = int(np.ceil(self.max_steps / self.current_episode_length))
                remaining_space = self.buffer_size - self.pos
                N = min(N, remaining_space // self.current_episode_length)
                print('pos is ', self.pos)
                print('padding times is ', N)
                print('current episode length is ', self.current_episode_length)
                # Create slices for the last episode's data
                obs_slice = self.observations[self.pos - self.current_episode_length: self.pos]
                action_slice = self.actions[self.pos - self.current_episode_length: self.pos]
                rewards_slice = self.rewards[self.pos - self.current_episode_length: self.pos]
                episode_start_slice = self.episode_starts[self.pos - self.current_episode_length: self.pos]
                values_slice = self.values[self.pos - self.current_episode_length: self.pos]
                log_probs_slice = self.log_probs[self.pos - self.current_episode_length: self.pos]
                # print('obs is ', self.observations)
                # print('obs slice is ', obs_slice)

                # Repeat the last episode's data N times along the second dimension
                for i in range(N):
                    self.observations[self.pos:self.pos + self.current_episode_length] = np.tile(obs_slice, 1)
                    self.actions[self.pos:self.pos + self.current_episode_length] = np.tile(action_slice, 1)
                    self.rewards[self.pos:self.pos + self.current_episode_length] = np.tile(rewards_slice, 1)
                    self.episode_starts[self.pos:self.pos + self.current_episode_length] = np.tile(episode_start_slice, 1)  # 1D, so no change here
                    self.values[self.pos:self.pos + self.current_episode_length] = np.tile(values_slice, 1)
                    self.log_probs[self.pos:self.pos + self.current_episode_length] = np.tile(log_probs_slice, 1)
                    self.pos += self.current_episode_length
                if self.pos == self.buffer_size:
                    self.full = True

                # Update position
                # self.pos += self.current_episode_length
                # # Repeat trajectory N times
                # self.observations = np.tile(self.observations[-self.current_episode_length:], (N, 1))[:self.max_steps]
                # self.actions = np.tile(self.actions[-self.current_episode_length:], (N, 1))[:self.max_steps]
                # self.rewards = np.tile(self.rewards[-self.current_episode_length:], (N,))[:self.max_steps]
                # self.episode_starts = np.tile(self.episode_starts[-self.current_episode_length:], (N,))[:self.max_steps]
                # # Optionally repeat values, log_probs, etc.
                # self.values = np.tile(self.values[-self.current_episode_length:], (N, 1))[:self.max_steps]
                # self.log_probs = np.tile(self.log_probs[-self.current_episode_length:], (N, 1))[:self.max_steps]
                # self.pos += self.current_episode_length * N
                # print('obs 1 N is ', self.observations[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('obs 2 N is ', self.observations[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('obs 3 N is ', self.observations[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
                # print('actions 1 N ', self.actions[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('actions 2 N ', self.actions[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('actions 3 N ', self.actions[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
                # print('rewards 1 N ', self.rewards[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('rewards 2 N ', self.rewards[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('rewards 3 N ', self.rewards[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
                # print('episode_starts 1 N ', self.episode_starts[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('episode_starts 2 N ', self.episode_starts[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('episode_starts 3 N ', self.episode_starts[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
                # print('values 1 N ', self.values[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('values 2 N ', self.values[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('values 3 N ', self.values[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
                # print('log probs 1 N ', self.log_probs[self.pos - self.current_episode_length * 1: self.pos - self.current_episode_length * 0])
                # print('log probs 2 N ', self.log_probs[self.pos - self.current_episode_length * 2: self.pos - self.current_episode_length * 1])
                # print('log probs 3 N ', self.log_probs[self.pos - self.current_episode_length * 3: self.pos - self.current_episode_length * 2])
        self.flag = False

    def log_collisions(self):
        self.flag = True

    def return_pos(self):
        return self.pos

    def reset(self):
        super().reset()
        self.current_episode_length = 0


class OnPolicyAdversarialAlgorithm(OnPolicyAlgorithm):
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
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if self.fni_flag:
                    actions, std, _action = self.trained_agent(obs_tensor[:, :-2])
                    actions = actions.cpu().numpy()
                else:
                    actions, _states = self.trained_agent.predict(obs_tensor[:, :-2].cpu(), deterministic=True)
                actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
                obs_tensor[:, -1] = (actions_tensor + 1) / 2
                adv_actions, adv_values, adv_log_probs = self.policy(obs_tensor)


            adv_actions = adv_actions.cpu().numpy()
            # Rescale and perform action
            clipped_adv_actions = adv_actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    # clipped_actions = self.policy.unscale_action(clipped_actions)
                    clipped_adv_actions = self.policy.unscale_action(clipped_adv_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    # clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
                    clipped_adv_actions = np.clip(adv_actions, self.action_space.low, self.action_space.high)

            adv_action_mask = (clipped_adv_actions[:, 0] > 0) & (obs_tensor[:, -2].cpu().numpy() > 0)
            if adv_action_mask or args.unlimited_attack:
                if args.env_name == 'TrafficEnv5-v0':
                    if args.attack_method == 'fgsm':
                        adv_state = FGSM_v2(clipped_adv_actions[:, 1:], victim_agent=self.trained_agent,
                                            last_state=obs_tensor[:, :-2], device=self.device, epsilon=args.attack_eps)
                    elif args.attack_method == 'fabfgsm':
                        adv_state = FAB_FGSM_v2(self.trained_agent, clipped_adv_actions[:, 1:], obs_tensor[:, :-2],
                                                device=self.device)
                    elif args.attack_method == 'pgd':
                        adv_state = PGD(clipped_adv_actions[:, 1:], self.trained_agent, obs_tensor[:, :-2],
                                        device=self.device, epsilon=args.attack_eps)
                    elif args.attack_method == 'cw':
                        adv_state = cw_attack_v2(self.trained_agent, obs_tensor[:, :-2], clipped_adv_actions[:, 1:])

                    if args.attack_method == 'direct':
                        action = clipped_adv_actions[:, 1:]
                    else:
                        if self.fni_flag:
                            adv_action_fromState, _, _ = self.trained_agent(adv_state)
                            action = adv_action_fromState.detach().cpu().numpy()
                        else:
                            adv_action_fromState, _states = self.trained_agent.predict(adv_state.cpu(),
                                                                                       deterministic=True)
                            action = adv_action_fromState
                else:
                    if args.attack_method == 'fgsm':
                        adv_state = FGSM_v2(clipped_adv_actions[:, 1], victim_agent=self.trained_agent,
                                                last_state=obs_tensor[:, :-2], device=self.device,epsilon=args.attack_eps)
                    elif args.attack_method == 'fabfgsm':
                        adv_state = FAB_FGSM_v2(self.trained_agent, clipped_adv_actions[:, 1], obs_tensor[:, :-2], device=self.device)
                    elif args.attack_method == 'pgd':
                        adv_state = PGD(clipped_adv_actions[:, 1], self.trained_agent, obs_tensor[:, :-2], device=self.device,epsilon=args.attack_eps)
                    elif args.attack_method == 'cw':
                        adv_state = cw_attack_v2(self.trained_agent, obs_tensor[:, :-2], clipped_adv_actions[:, 1])

                    if args.attack_method == 'direct':
                        action = clipped_adv_actions[:, 1]
                    else:
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
            # rewards = -1 * rewards

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

            if isinstance(rollout_buffer, CustomRolloutBuffer) and infos[0]['flag']:
                rollout_buffer.log_collisions()
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                adv_actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                adv_values,
                adv_log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            if isinstance(rollout_buffer, CustomRolloutBuffer) and infos[0]['flag']:
                n_steps = rollout_buffer.return_pos()
            # print('last episode start is ', self._last_episode_starts)

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _dump_logs(self, iteration):
        super()._dump_logs(iteration)
        if safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]) >= self.max_epi_reward:
            self.save(self.best_model_path)
            self.max_epi_reward = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])



class AdversarialPPO(OnPolicyAdversarialAlgorithm, PPO):
    def __init__(self, algo, path, aargs,env_name, best_model_path,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instantiate the agent
        model_path = os.path.join(path, env_name, algo, 'lunar')
        if algo == 'PPO':
            self.trained_agent = PPO.load(model_path, device=self.device)
        elif algo == 'SAC':
            self.trained_agent = SAC.load(model_path, device=self.device)
        elif algo == 'TD3':
            self.trained_agent = TD3.load(model_path, device=self.device)

        self.max_epi_reward = 0
        self.best_model_path = best_model_path
        self.fni_flag = True if algo == 'FNI' else False

        #for expert prior
        self.aargs = aargs
        if self.aargs.expert_attack:
            if self.aargs.expert_prior == 'ValuePenalty':
                self.alpha = 0.01
            elif self.aargs.expert_prior == 'PolicyConstrained':
                self.alpha = 0.01
                #在对数空间内优化
                self.lambda_raw = th.nn.Parameter(
                    th.tensor(np.log(self.alpha), dtype=th.float32, device=self.device)
                )
                self.lambda_optimizer = th.optim.Adam([self.lambda_raw], lr=1e-4)
        self.epsilon = 0.5


        # 加载专家模型
        if aargs.expert_attack:
            # 专家模型目录
            if aargs.expert_model_path == '':
                expert_model_path = f'./expert_model/{aargs.env_name}_{aargs.algo}_{aargs.adv_algo}_{aargs.expert_train_samples}'
            self.expert_models = load_ensemble_models(aargs.expert_model_path, (28,), 2, device)

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
        exp_kls = []
        lambda_vals,  lambda_losses = [], []
        exp_miu, exp_v = [],[]
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
                    expert_actions,expert_means,expert_vars =sample_from_mixture(obs,self.expert_models,device=self.device)
                    exp_miu.append(expert_means.mean().item())
                    exp_v.append(expert_vars.mean().item())

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                if self.aargs.expert_attack:
                    #计算专家分布和当前动作分布的kl散度
                    current_dist = self.policy.get_distribution(obs).distribution
                    expert_dist = th.distributions. normal. Normal(expert_means, torch.sqrt(expert_vars))
                    exp_kl = dist.kl_divergence(current_dist, expert_dist)
                    exp_kl = torch.mean(exp_kl, dim=1)
                    exp_kls.append(exp_kl.mean().item())
                    #print(exp_kl.shape)

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
                    policy_loss = (-th.min(policy_loss_1, policy_loss_2) + self.alpha * exp_kl).mean()
                elif self.aargs.expert_attack and self.aargs.expert_prior == 'PolicyConstrained':
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
                    value_loss = F.mse_loss(rollout_data.returns - self.alpha * exp_kl, values_pred )
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

                if self.aargs.expert_attack and self.aargs.expert_prior == 'PolicyConstrained':
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
            if self.aargs.expert_prior == 'PolicyConstrained':
                self.logger.record("train/lambda_val", np.mean(lambda_vals))
                self.logger.record("train/lambda_loss", np.mean(lambda_losses))

# get parameters from config.py
parser = get_config()
args = parser.parse_args()

# Create environment
env = gym.make(args.env_name, attack=args.attack, adv_steps=args.adv_steps)
env = TimeLimit(env, max_episode_steps=args.T_horizon)
env = Monitor(env)
env.unwrapped.start()


if args.expert_attack:
    expert_msg = f'ens2_{args.env_name}_{args.algo}_{args.expert_prior}_eps{args.attack_eps}_{args.adv_steps}'
else:
    if args.addition_msg == '':
        expert_msg = f'{args.env_name}_{args.algo}_NoExpert_eps{args.attack_eps}'
    else:
        expert_msg = args.addition_msg

# log path
eval_log_path = "./logs/adv_eval/" + os.path.join(args.adv_algo, args.env_name, args.algo, expert_msg)
os.makedirs(eval_log_path, exist_ok=True)
# eval_callback = EvalCallback(env,
#                              best_model_save_path=eval_log_path + "/best_model",
#                              log_path=eval_log_path,  # 日志保存路径
#                              eval_freq=300,  # 每150步评估一次
#                              n_eval_episodes=5,  # 每次评估5个episode
#                              deterministic=True)
best_model_path = os.path.join(eval_log_path, "best_model")
model_path = os.path.join(eval_log_path, 'model')
os.makedirs(model_path, exist_ok=True)

# 设置设备
if args.use_cuda and th.cuda.is_available():
    device = th.device(f"cuda:{args.cuda_number}")
else:
    device = th.device("cpu")

# 设置随机种子
random.seed(args.seed)  # 设置 Python 随机种子
np.random.seed(args.seed)  # 设置 NumPy 随机种子
th.manual_seed(args.seed)  # 设置 CPU 随机种子
if th.cuda.is_available():
    th.cuda.manual_seed(args.seed)  # 设置 CUDA 随机种子
    th.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 随机种子
th.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
th.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化

checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=model_path)

# whether padding
if args.padding:
    rollout_buffer_class = CustomRolloutBuffer
else:
    rollout_buffer_class = RolloutBuffer


# init wandb
if not args.no_wandb:
    run_name = f"{args.attack_method}-{args.algo}-{expert_msg}"
    run = wandb.init(project="ExpertPriorRL", name=run_name, config=args, sync_tensorboard=True)
    model = AdversarialPPO(args.algo, args.path, args,args.env_name, best_model_path, "MlpPolicy", env, n_steps=args.n_steps, verbose=1,
                           tensorboard_log=f"runs/{run.id}", rollout_buffer_class=rollout_buffer_class, device=device)
    wandb_callback = WandbCallback(gradient_save_freq=5, verbose=2, model_save_path=f"models/{run.id}",)
    model.learn(total_timesteps=args.train_step*args.n_steps, progress_bar=True, callback=[checkpoint_callback, wandb_callback])
else:
    model = AdversarialPPO(args.algo, args.path, args,args.env_name, best_model_path, "MlpPolicy", env, n_steps=args.n_steps, verbose=1, rollout_buffer_class=rollout_buffer_class, device=device)
    model.learn(total_timesteps=args.train_step * args.n_steps, progress_bar=True,
                callback=checkpoint_callback)
# 读取评估日志文件
# eval_log_file = os.path.join(eval_log_path, "evaluations.npz")

# 绘制评估奖励曲线
plt.plot(env.get_episode_rewards())
plt.title('Rewards per episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(os.path.join(eval_log_path, "rewards.png"), dpi=300)
plt.close()

plt.plot(env.get_episode_lengths())
plt.title('Steps per episode')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.savefig(os.path.join(eval_log_path, "steps.png"), dpi=300)
plt.close()

reward_df = pd.DataFrame(env.get_episode_rewards())
step_df = pd.DataFrame(env.get_episode_lengths())
reward_df.to_csv(os.path.join(eval_log_path, "rewards.csv"), index=False)
step_df.to_csv(os.path.join(eval_log_path, "steps.csv"), index=False)
# Save the log

# Save the agent
model.save(os.path.join(eval_log_path, "lunar"))
del model  # delete trained model to demonstrate loading
env.close()
