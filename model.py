"""
@Time ： 2021/2/7 21:10
@Auth ： Duan Yihai
@File ：model.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MuNet(nn.Module):
    def __init__(self, state_dim):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc3(x))*2



class DDPG_QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPG_QNet, self).__init__()
        self.fc_s = nn.Linear(state_dim, 64)
        self.fc_a = nn.Linear(action_dim, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, s, a):
        s = F.relu(self.fc_s(s))
        a = F.relu(self.fc_a(a))
        cat = torch.cat([s, a], dim=1)
        x = F.relu(self.fc_q(cat))
        x = self.fc_out(x)
        return x

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

