"""
@Time ： 2021/2/7 15:13
@Auth ： Duan Yihai
@File ：dqn.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""

import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import argparse
import numpy as np
from utils import loadLogger
from memory import ReplayBuffer
from utils import str2bool
import tensorboard_easy as te
from dqn_policy import DQN
import configparser
import datetime

def get_args():
    parser = argparse.ArgumentParser('Shenhai RL arguments for DQN')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='Environment name')
    parser.add_argument('--epoch', type=int, default=10000, help='Number of epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--batch_size', type=int, default=32, help='seed')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Replay Buffer size')
    parser.add_argument('--update_interval', type=int, default=20, help='update qnet_target interval')
    parser.add_argument('--load_model', type=str2bool, default='false', help='weather load model to continue')
    parser.add_argument('--info_dir', type=str, default='info/dqn')
    parser.add_argument('--te_dir', type=str, default='te/dqn')
    parser.add_argument('--config_name', type=str, default='config.conf')
    args = parser.parse_args()
    return args

def eval_policy(num, env_name, policy):
    rewards = []
    env = gym.make(env_name)
    for i in range(num):
        s = env.reset()
        done = False
        reward = 0
        while not done:
            a = policy.select_action(torch.from_numpy(s).float(), eval=True)
            s, r, done, info = env.step(a)
            reward += r
            if done:
                rewards.append(reward)
                reward = 0
                s = env.reset()

    return np.mean(rewards)

def main():
    args = get_args()

    info_logger = loadLogger(args.info_dir)
    info_logger.info(args)
    nowtime = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    te_board = te.Logger(os.path.join(args.te_dir, str(args.seed) + '_' + nowtime))

    cf = configparser.ConfigParser()
    cf.read(args.config_name)
    default_config = cf['DEFAULT']

    # TODO: Initialization
    # 1. Initialize environment
    env = gym.make(args.env)
    print(env.action_space.n, env.observation_space.shape[0])

    # 2. Initialize DQN Policy
    dqn = DQN()

    # 3. Initialize Replay Buffer
    memory = ReplayBuffer(args.buffer_size)


    step = 0
    rewards = []

    for i_epoch in range(args.epoch):
        s = env.reset()
        done = False
        reward = 0

        # TODO: Rollout
        while not done:
            action = dqn.select_action(torch.from_numpy(s).float())
            s_next, r, done, info = env.step(action)
            transition = (s, action, r/100.0, s_next, done)
            memory.add(transition)
            s = s_next
            reward += r
            step += 1
            if done:
                te_board.log_scalar('train/reward', value=reward, step=step)
                rewards.append(reward)
                reward = 0
                s = env.reset()

        # TODO: Train
        if memory.size() > int(default_config['start_train']):
            loss, iterations, epsilon = dqn.train(memory)
            te_board.log_scalar('episode', epsilon, iterations)
            te_board.log_scalar('train/loss', loss, step)

        # TODO: Test
        if i_epoch % 30 == 0 and i_epoch > 0:
            eval_reward = eval_policy(20, args.env, dqn)
            te_board.log_scalar('test/reward', value=eval_reward, step=step)

        # TODO: Update Qnet_target
        if i_epoch % args.update_interval == 0 and i_epoch != 0:
            print('Epoch:{} Reward:{}, Buffer Size:{}'.format(i_epoch, np.mean(rewards[-20:]), memory.size()))

if __name__=='__main__':
    main()