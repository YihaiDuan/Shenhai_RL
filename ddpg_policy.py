"""
@Time ： 2021/2/22 18:23
@Auth ： Duan Yihai
@File ：ddpg_policy.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""

"""
@Time ： 2021/2/22 16:07
@Auth ： Duan Yihai
@File ：dqn_policy.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model import MuNet, DDPG_QNet, OrnsteinUhlenbeckNoise
import random

class DDPG(object):
    def __init__(
            self,
            action_dim=1,
            state_dim=4,
            gamma=0.98,
            optimizer='Adam',
            lr_mu=0.0005,
            lr_q=0.001,
            batch_size=32,
            target_update_frequency=20,
            soft_update=True,
            tau=0.005
    ):
        self.q, self.q_target = DDPG_QNet(state_dim, action_dim), DDPG_QNet(state_dim, action_dim)
        self.mu, self.mu_target = MuNet(state_dim), MuNet(state_dim)
        self.q_target.load_state_dict(self.q.state_dict())
        self.mu_target.load_state_dict(self.mu.state_dict())
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

        self.gamma = gamma
        self.batch_size = batch_size

        if optimizer == 'Adam':
            self.mu_optimizer = optim.Adam(params=self.mu.parameters(), lr=lr_mu)
            self.q_optimizer = optim.Adam(params=self.q.parameters(), lr=lr_q)


        # update method
        self.update_target = self.soft_update if soft_update else self.copy_update
        self.tau = tau
        self.target_update_frequency = target_update_frequency

        self.iterations = 0

    def train(self, buffer):
        mini_batch = buffer.sample(self.batch_size)
        s, a, r, s_next, done = mini_batch

        target = r + self.gamma * self.q_target(s_next, self.mu_target(s_next)) * done
        q_loss = F.smooth_l1_loss(self.q(s, a), target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        mu_loss = -self.q(s, self.mu(s)).mean()
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()


        self.update_target()

        self.iterations += 1

        return q_loss, mu_loss, self.iterations, self.epsilon

    def select_action(self, state, eval=False):
        out = self.Q(state)
        self.epsilon = self.eval_eps if eval else max(self.end_eps, self.initial_eps - self.iterations * self.slope)

        coin = random.random()
        if coin < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            return out.argmax().item()

    def copy_update(self):
        raise NotImplementedError

    def soft_update(self):
        for param, target_param in zip(self.q.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.mu.parameters(), self.mu_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

