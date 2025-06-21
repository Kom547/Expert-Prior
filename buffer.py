from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces
import numpy as np
from typing import NamedTuple
import torch as th

class NewRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class DecoupleRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer with only decouple policy update
    """

    def __init__(self, buffer_size, observation_space, action_space, device, gamma=0.99, gae_lambda=1, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma, gae_lambda, n_envs)

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, self.action_dim)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = np.array(log_prob.clone().cpu())
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def return_pos(self):
        return self.pos

    def reset(self):
        super().reset()
        self.current_episode_length = 0
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)

    def get(self, batch_size):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds, env=None):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds],
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return NewRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class PaddingRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer with only trajectory padding
    """

    def __init__(self, buffer_size, observation_space, action_space, device, gamma=0.99, gae_lambda=1, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma, gae_lambda, n_envs)
        self.max_steps = 30  # Assuming buffer_size as max_steps
        self.current_episode_length = 1
        self.flag = False

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        # print('episode_starts is ', self.episode_starts[self.pos-1])
        if self.episode_starts[self.pos - 1][0]:
            self.current_episode_length = 1
        else:
            self.current_episode_length += 1
        # Check if the episode has ended
        if self.flag:
            print('*****************************padding***************************************')
            # If episode length < max_steps, repeat the trajectory
            if self.current_episode_length < self.max_steps and self.pos >= self.current_episode_length:
                N = int(np.ceil(self.max_steps / self.current_episode_length))
                remaining_space = self.buffer_size - self.pos
                print('remaining_space', remaining_space)
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

                # Repeat the last episode's data N times along the second dimension
                for i in range(N):
                    self.observations[self.pos:self.pos + self.current_episode_length] = np.tile(obs_slice, 1)
                    self.actions[self.pos:self.pos + self.current_episode_length] = np.tile(action_slice, 1)
                    self.rewards[self.pos:self.pos + self.current_episode_length] = np.tile(rewards_slice, 1)
                    self.episode_starts[self.pos:self.pos + self.current_episode_length] = np.tile(episode_start_slice,
                                                                                                   1)  # 1D, so no change here
                    self.values[self.pos:self.pos + self.current_episode_length] = np.tile(values_slice, 1)
                    self.log_probs[self.pos:self.pos + self.current_episode_length] = np.tile(log_probs_slice, 1)
                    self.pos += self.current_episode_length
                if self.pos == self.buffer_size:
                    self.full = True
        self.flag = False

    def log_collisions(self):
        self.flag = True

    def return_pos(self):
        return self.pos


class DecouplePaddingRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer with trajectory padding and decouple policy update
    """

    def __init__(self, buffer_size, observation_space, action_space, device, gamma=0.99, gae_lambda=1, n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device, gamma, gae_lambda, n_envs)
        self.max_steps = 30  # Assuming buffer_size as max_steps
        self.current_episode_length = 1
        self.flag = False

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, self.action_dim)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = np.array(log_prob.clone().cpu())
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        # print('episode_starts is ', self.episode_starts[self.pos-1])
        if self.episode_starts[self.pos - 1][0]:
            self.current_episode_length = 1
        else:
            self.current_episode_length += 1
        # Check if the episode has ended
        if self.flag:
            # print('*****************************padding***************************************')
            # If episode length < max_steps, repeat the trajectory
            if self.current_episode_length < self.max_steps and self.pos >= self.current_episode_length:
                N = int(np.ceil(self.max_steps / self.current_episode_length))
                remaining_space = self.buffer_size - self.pos
                N = min(N, remaining_space // self.current_episode_length)
                # print('pos is ', self.pos)
                # print('padding times is ', N)
                # print('current episode length is ', self.current_episode_length)
                # Create slices for the last episode's data
                obs_slice = self.observations[self.pos - self.current_episode_length: self.pos]
                action_slice = self.actions[self.pos - self.current_episode_length: self.pos]
                rewards_slice = self.rewards[self.pos - self.current_episode_length: self.pos]
                episode_start_slice = self.episode_starts[self.pos - self.current_episode_length: self.pos]
                values_slice = self.values[self.pos - self.current_episode_length: self.pos]
                log_probs_slice = self.log_probs[self.pos - self.current_episode_length: self.pos]
                # print('obs is ', self.observations)
                # print('obs slice ', obs_slice)
                # print('obs slice is ', obs_slice)

                # Repeat the last episode's data N times along the second dimension
                for i in range(N):
                    self.observations[self.pos:self.pos + self.current_episode_length] = np.tile(obs_slice, 1)
                    self.actions[self.pos:self.pos + self.current_episode_length] = np.tile(action_slice, 1)
                    self.rewards[self.pos:self.pos + self.current_episode_length] = np.tile(rewards_slice, 1)
                    self.episode_starts[self.pos:self.pos + self.current_episode_length] = np.tile(episode_start_slice,
                                                                                                   1)  # 1D, so no change here
                    self.values[self.pos:self.pos + self.current_episode_length] = np.tile(values_slice, 1)
                    self.log_probs[self.pos:self.pos + self.current_episode_length] = np.tile(log_probs_slice, 1)
                    self.pos += self.current_episode_length
                if self.pos == self.buffer_size:
                    self.full = True

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
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)

    def _get_samples(self, batch_inds, env=None):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds],
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))