import random
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler

class RolloutStorage(object):
    def __init__(self, config):
        self.obs = torch.zeros([config.max_buff,  *config.state_shape], dtype=torch.uint8)
        self.next_obs = torch.zeros([config.max_buff,  *config.state_shape], dtype=torch.uint8)
        self.rewards = torch.zeros([config.max_buff,  1])
        self.actions = torch.zeros([config.max_buff, 1])
        self.actions = self.actions.long()
        self.masks = torch.ones([config.max_buff,  1])
        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.num_steps = config.max_buff
        self.step = 0
        self.current_size = 0

    def add(self, obs, actions, rewards, next_obs, masks):
        self.obs[self.step].copy_(torch.tensor(obs[None,:], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.next_obs[self.step].copy_(torch.tensor(next_obs[None,:], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.actions[self.step].copy_(torch.tensor(actions, dtype=torch.float))
        self.rewards[self.step].copy_(torch.tensor(rewards, dtype=torch.float))
        self.masks[self.step].copy_(torch.tensor(masks, dtype=torch.float))
        self.step = (self.step + 1) % self.num_steps
        self.current_size = min(self.current_size + 1, self.num_steps)

    def sample(self, mini_batch_size=None):
        indices = np.random.randint(0, self.current_size, mini_batch_size)
        obs_batch = self.obs[indices]
        obs_next_batch = self.next_obs[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        masks_batch = self.masks[indices]
        return obs_batch, obs_next_batch, actions_batch, rewards_batch, masks_batch
