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

        return torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), torch.tensor(r_list),\
               torch.tensor(s_next_list, dtype=torch.float), torch.tensor(done_list)

    def size(self):
        return len(self.buffer)