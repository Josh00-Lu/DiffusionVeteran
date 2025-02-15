from typing import Dict

import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils import GaussianNormalizer, dict_apply


class DV_D4RLAntmazeSeqDataset(BaseDataset):
    """ **D4RL-Antmaze Sequential Dataset**

        torch.utils.data.Dataset wrapper for D4RL-Antmaze dataset.
        Chunk the dataset into sequences of length `horizon` with obs-repeat/act-zero/reward-zero padding.
        Use GaussianNormalizer to normalize the observations as default.
        Each batch contains
        - batch["obs"]["state"], observations of shape (batch_size, horizon, o_dim)
        - batch["act"], actions of shape (batch_size, horizon, a_dim)
        - batch["rew"], rewards of shape (batch_size, horizon, 1)
        - batch["val"], Monte Carlo return of shape (batch_size, 1)

        Args:
            dataset: Dict[str, np.ndarray],
                D4RL-Antmaze dataset. Obtained by calling `env.get_dataset()`.
            horizon: int,
                Length of each sequence. Default is 1.
            max_path_length: int,
                Maximum length of the episodes. Default is 1001.
            noreaching_penalty: float,
                Penalty for not reaching the goal. Default is -100.
            discount: float,
                Discount factor. Default is 0.99.

        Examples:
            >>> env = gym.make("antmaze-medium-play-v2")
            >>> dataset = D4RLAntmazeDataset(env.get_dataset(), horizon=32)
            >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            >>> batch = next(iter(dataloader))
            >>> obs = batch["obs"]["state"]  # (32, 32, 29)
            >>> act = batch["act"]           # (32, 32, 8)
            >>> rew = batch["rew"]           # (32, 32, 1)
            >>> val = batch["val"]           # (32, 1)

            >>> normalizer = dataset.get_normalizer()
            >>> obs = env.reset()[None, :]
            >>> normed_obs = normalizer.normalize(obs)
            >>> unnormed_obs = normalizer.unnormalize(normed_obs)
        """
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            horizon: int = 1,
            max_path_length: int = 1001,
            discount: float = 0.99,
            continous_reward_at_done: bool = False,
            reward_tune: str = "iql",
            center_mapping: bool = True,
            learn_policy: bool = False,
            stride: int = 1,
            only_learn_reached_policy: bool = False,
    ):
        super().__init__()

        observations, actions, rewards, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"].astype(np.float32),
            dataset["terminals"].astype(np.float32))
        
        self.learn_policy = learn_policy
        self.stride = stride

        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        self.indices = []
        self.seq_obs, self.seq_act, self.seq_rew, self.seq_tml = [], [], [], []

        ptr = 0 # ptr: t=0 pos
        path_idx = 0

        timeouts_indices = np.where(timeouts == 1)[0]

        for index in timeouts_indices:
            terminal_index = -1
            assert index - ptr + 1 == max_path_length
            for i in range(ptr, index + 1):
                if terminals[i]:
                    terminal_index = i
                    break
            
            if terminal_index == -1 and self.learn_policy and not only_learn_reached_policy:
                path_length = index - ptr + 1
                assert path_length == max_path_length
                
                _seq_obs = np.zeros((max_path_length + (horizon - 1) * stride, self.o_dim), dtype=np.float32)
                _seq_act = np.zeros((max_path_length + (horizon - 1) * stride, self.a_dim), dtype=np.float32)
                _seq_rew = np.zeros((max_path_length + (horizon - 1) * stride, 1), dtype=np.float32)
                _seq_tml = np.zeros((max_path_length + (horizon - 1) * stride, 1), dtype=np.float32)
                
                _seq_obs[:path_length] = normed_observations[ptr:index+1]
                _seq_act[:path_length] = actions[ptr:index+1]
                _seq_rew[:path_length] = rewards[ptr:index+1][:, None]
                _seq_tml[:path_length] = terminals[ptr:index+1][:, None]
                
                self.seq_obs.append(_seq_obs)
                self.seq_act.append(_seq_act)
                self.seq_rew.append(_seq_rew)
                self.seq_tml.append(_seq_tml)
                
                max_start = max_path_length - (horizon - 1) * stride - 1
                self.indices += [(path_idx, start, start + (horizon - 1) * stride + 1) for start in range(max_start + 1)]
                path_idx += 1
                              
            elif terminal_index != -1:
                path_length = terminal_index - ptr + 1
                assert path_length <= max_path_length
                assert rewards[terminal_index] == 1
                
                _seq_obs = np.zeros((max_path_length + (horizon - 1) * stride, self.o_dim), dtype=np.float32)
                _seq_act = np.zeros((max_path_length + (horizon - 1) * stride, self.a_dim), dtype=np.float32)
                _seq_rew = np.zeros((max_path_length + (horizon - 1) * stride, 1), dtype=np.float32)
                _seq_tml = np.zeros((max_path_length + (horizon - 1) * stride, 1), dtype=np.float32)

                _seq_obs[:path_length] = normed_observations[ptr:terminal_index+1]
                _seq_act[:path_length] = actions[ptr:terminal_index+1]
                _seq_rew[:path_length] = rewards[ptr:terminal_index+1][:, None]
                _seq_tml[:path_length] = terminals[ptr:terminal_index+1][:, None]

                _seq_obs[path_length:] = normed_observations[terminal_index]  # repeat state
                _seq_act[path_length:] = 0 # zero action
                _seq_rew[path_length:] = 1 if continous_reward_at_done else 0
                _seq_tml[path_length:] = 1
                
                self.seq_obs.append(_seq_obs)
                self.seq_act.append(_seq_act)
                self.seq_rew.append(_seq_rew)
                self.seq_tml.append(_seq_tml)
            
                max_start = path_length - 1
                self.indices += [(path_idx, start, start + (horizon - 1) * stride + 1) for start in range(max_start + 1)]
                path_idx += 1
            
            ptr = index + 1

        self.seq_obs = np.array(self.seq_obs)
        self.seq_act = np.array(self.seq_act)
        self.seq_rew = np.array(self.seq_rew)
        self.seq_tml = np.array(self.seq_tml)
        
        if reward_tune == "iql":
            self.seq_rew += -1 
        elif reward_tune == "none":
            self.seq_rew = self.seq_rew
        else:
            raise ValueError(f"reward_tune: {reward_tune} is not supported.")

        self.seq_val = np.copy(self.seq_rew)
        for i in reversed(range(max_path_length - 1)):
            self.seq_val[:, i] = self.seq_rew[:, i] + discount * self.seq_val[:, i+1]
        
        print(f"max discounted return: {self.seq_val.max()}")
        print(f"min discounted return: {self.seq_val.min()}")
        
        # val \in [-1, 1]
        self.seq_val = (self.seq_val - self.seq_val.min()) / (self.seq_val.max() - self.seq_val.min())
        if center_mapping:
            self.seq_val = self.seq_val * 2 - 1
        print(f"max normed discounted return: {self.seq_val.max()}")
        print(f"min normed discounted return: {self.seq_val.min()}")

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]
        
        if self.learn_policy:
            horizon_state = self.seq_obs[path_idx, start:end:self.stride].copy()
            horizon_state[:, :2] -= horizon_state[0, :2]
        else:
            horizon_state = self.seq_obs[path_idx, start:end:self.stride]

        data = {
            'obs': {'state': horizon_state},
            'act': self.seq_act[path_idx, start:end:self.stride],
            'rew': self.seq_rew[path_idx, start:end:self.stride],
            'val': self.seq_val[path_idx, start],
            'tml': self.seq_tml[path_idx, start],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data


class D4RLAntmazeTDDataset(BaseDataset):
    """ **D4RL-Antmaze Transition Dataset**

    torch.utils.data.Dataset wrapper for D4RL-Antmaze dataset.
    Chunk the dataset into transitions.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    - batch["obs"]["state"], observation of shape (batch_size, o_dim)
    - batch["next_obs"]["state"], next observation of shape (batch_size, o_dim)
    - batch["act"], action of shape (batch_size, a_dim)
    - batch["rew"], reward of shape (batch_size, 1)
    - batch["tml"], terminal of shape (batch_size, 1)

    Args:
        dataset: Dict[str, np.ndarray],
            D4RL-MuJoCo TD dataset. Obtained by calling `d4rl.qlearning_dataset(env)`.
        reward_tune: str,
            Reward tuning method. Can be "iql", "cql", "antmaze", or "none". Default is "iql".

    Examples:
        >>> env = gym.make("antmaze-medium-play-v2")
        >>> dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env))
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["obs"]["state"]  # (32, 29)
        >>> act = batch["act"]           # (32, 8)
        >>> rew = batch["rew"]           # (32, 1)
        >>> tml = batch["tml"]           # (32, 1)
        >>> next_obs = batch["next_obs"]["state"]  # (32, 29)

        >>> normalizer = dataset.get_normalizer()
        >>> obs = env.reset()[None, :]
        >>> normed_obs = normalizer.normalize(obs)
        >>> unnormed_obs = normalizer.unnormalize(normed_obs)
    """
    def __init__(self, dataset: Dict[str, np.ndarray], reward_tune: str = "iql"):
        super().__init__()

        observations, actions, next_observations, rewards, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["next_observations"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["terminals"].astype(np.float32))

        if reward_tune == "iql":
            rewards = rewards - 1.
        elif reward_tune == "cql":
            rewards = (rewards - 0.5) * 4.
        elif reward_tune == "antmaze":
            rewards = (rewards - 0.25) * 2.
        elif reward_tune == "none":
            rewards = rewards
        else:
            raise ValueError(f"reward_tune: {reward_tune} is not supported.")

        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)
        normed_next_observations = self.normalizers["state"].normalize(next_observations)

        self.obs = torch.tensor(normed_observations)
        self.act = torch.tensor(actions)
        self.rew = torch.tensor(rewards)[:, None]
        self.tml = torch.tensor(terminals)[:, None]
        self.next_obs = torch.tensor(normed_next_observations)

        self.size = self.obs.shape[0]
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):

        data = {
            'obs': {
                'state': self.obs[idx], },
            'next_obs': {
                'state': self.next_obs[idx], },
            'act': self.act[idx],
            'rew': self.rew[idx],
            'tml': self.tml[idx], }

        return data