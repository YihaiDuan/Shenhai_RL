"""
@Time ： 2021/2/7 20:10
@Auth ： Duan Yihai
@File ：just_test.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""
import collections
import random
import torch
a = collections.deque(maxlen=10)

a = torch.tensor([[2,1,3], [2, 5, 3]])
print(a.max(1)[0])