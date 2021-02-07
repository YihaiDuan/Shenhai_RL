"""
@Time ： 2021/2/7 20:10
@Auth ： Duan Yihai
@File ：just_test.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""
import collections
import random
a = collections.deque(maxlen=10)

for i in range(15):
    a.append(i)

b = random.sample(a, 3)
print(b)