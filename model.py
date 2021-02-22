"""
@Time ： 2021/2/7 21:10
@Auth ： Duan Yihai
@File ：model.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""

import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def select_action(self, input, epsilon, test=False):
    #     out = self.forward(input)
    #     if test:
    #         return out.argmax().item()
    #     else:
    #         coin = random.random()
    #         if coin < epsilon:
    #             return random.randint(0, 1)
    #         else:
    #             return out.argmax().item()
