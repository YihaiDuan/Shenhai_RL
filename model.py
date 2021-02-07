"""
@Time ： 2021/2/7 21:10
@Auth ： Duan Yihai
@File ：model.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""

import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        pass