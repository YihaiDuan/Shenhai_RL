"""
@Time ： 2021/4/30 16:00
@Auth ： Duan Yihai
@File ：test_gym.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""

import gym
env = gym.make('BreakoutNoFrameskip-v4')
env = gym.make('CartPole-v0')

env.reset()