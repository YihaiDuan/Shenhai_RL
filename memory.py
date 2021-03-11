"""
@Time ： 2021/2/7 14:45
@Auth ： Duan Yihai
@File ：memory.py
@IDE ：PyCharm
@Motto：Keep Coding, Keep Thinking
"""
import torch
import numpy as np
import random

import collections


class ReplayBuffer():
    def __init__(self, size):
        self.buffer = collections.deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        s_list, a_list, r_list, s_next_list, done_list = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_next, done = transition
            done = 0.0 if done else 1.0
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_next_list.append(s_next)
            done_list.append([done])

        return torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), torch.tensor(r_list), \
               torch.tensor(s_next_list, dtype=torch.float), torch.tensor(done_list)

    def size(self):
        return len(self.buffer)


class TorchReplayBuffer():
    def __init__(self, buffer_size, obs_shape, action_space):
        self.buffer_size = buffer_size
        self.obs = torch.zeros(self.buffer_size, *obs_shape)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(self.buffer_size, action_shape, dtype=torch.long)
        self.rewards = torch.zeros(self.buffer_size, 1)
        self.obs_next = torch.zeros(self.buffer_size, *obs_shape)
        self.done_mask = torch.zeros(self.buffer_size, 1)

        self.cur_size = 0
        self.idx = 0

    def add(self, obs, actions, rewards, obs_next, done_mask):
        self.obs[self.idx].copy_(obs)
        self.actions[self.idx].copy_(actions)
        self.rewards[self.idx].copy_(rewards)
        self.obs_next[self.idx].copy_(obs_next)
        self.done_mask[self.idx].copy_(done_mask)
        self.idx = (self.idx + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)

    def sample(self, batch_size=32):
        idx = torch.randint(0, self.cur_size, [batch_size])
        obs_batch = self.obs[idx]
        actions_batch = self.actions[idx]
        rewards_batch = self.rewards[idx]
        ob_next_batch = self.obs_next[idx]
        done_mask_batch = self.done_mask[idx]

        return obs_batch, actions_batch, rewards_batch, ob_next_batch, done_mask_batch

    def size(self):
        return self.cur_size
