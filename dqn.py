"""
@Time ： 2021/2/7 15:13
@Auth ： Duan Yihai
@File ：dqn.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""

import torch
import os
import gym
import argparse
import numpy as np
from utils import loadLogger
from memory import ReplayBuffer
from utils import str2bool
import tensorboard_easy as te

def get_args():
    parser = argparse.ArgumentParser('Shenhai RL arguments for DQN')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='Environment name')
    parser.add_argument('--epoch', type=int, default=10000, help='Number of epoch')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Replay Buffer size')
    parser.add_argument('--load_model', type=str2bool, default='false', help='weather load model to continue')
    parser.add_argument('--info_dir', type=str, default='info/dqn')
    parser.add_argument('--te_dir', type=str, default='te/dqn')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    memory = ReplayBuffer(args.buffer_size)
    info_logger = loadLogger(args.info_dir)
    info_logger.info(args)
    te_board = te.Logger(args.te_dir)

    env = gym.make(args.env)
    s = env.reset()
    print(env.action_space.n, env.observation_space.shape[0])
    for i in range(100):
        a = env.action_space.sample()
        s_next, r, done, info = env.step(a)
        transition = (s, a, r, s_next, done)
        memory.add(transition)
        s = s_next
        if done:
            s = env.reset()





if __name__=='__main__':
    main()