"""
@Time ： 2021/2/22 16:07
@Auth ： Duan Yihai
@File ：dqn_policy.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
import random

class DQN(object):
    def __init__(
            self,
            num_actions=2,
            state_dim=4,
            gamma=0.98,
            optimizer='Adam',
            lr=0.0005,
            batch_size=32,
            target_update_frequency=20,
            soft_update=False,
            tau=0.005,
            initial_eps=0.1,
            end_eps=0.01,
            eps_decay_period=1500,
            eval_eps=0.0
    ):
        self.num_actions = num_actions

        self.Q = QNet(state_dim, num_actions)
        self.Q_target = QNet(state_dim, num_actions)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.gamma = gamma
        self.batch_size = batch_size

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(params=self.Q.parameters(), lr=lr)

        # update method
        self.update_target = self.soft_update if soft_update else self.copy_update
        self.tau = tau
        self.target_update_frequency = target_update_frequency

        # decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.initial_eps - self.end_eps)/eps_decay_period
        self.eval_eps = eval_eps

        self.iterations = 0

    def train(self, buffer):
        mini_batch = buffer.sample(self.batch_size)
        s, a, r, s_next, done = mini_batch

        q_a = self.Q(s).gather(1, a)  # [batch_size, 1]

        max_q_next = self.Q_target(s_next).max(1)[0].unsqueeze(1)  # [batch_size, 1]
        target = r + self.gamma * max_q_next * done
        target = target.detach()

        loss = F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target()

        self.iterations += 1

        return loss, self.iterations, self.epsilon

    def select_action(self, state, eval=False):
        out = self.Q(state)
        self.epsilon = self.eval_eps if eval else max(self.end_eps, self.initial_eps - self.iterations * self.slope)

        coin = random.random()
        if coin < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            return out.argmax().item()

    def copy_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def soft_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

