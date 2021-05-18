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
import gym
# a = collections.deque(maxlen=10)
#
# env_name = 'CartPole-v0'
# env_name = 'FetchPush-v1'
#
# env = gym.make(env_name)
# s = env.reset()
# print(s)
# print(env.action_space)
# print(env.observation_space)
from PIL import Image
import matplotlib.pyplot as plt
import cdt
a = torch.zeros(3, dtype=torch.long)
print(a)
print(cdt.utils.Settings.SETTINGS.get_default(rpath=None))
# from torchvision import datasets, transforms
# trans_f = transforms.Compose([
#             transforms.CenterCrop(128),
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
# train_set = datasets.CelebA('back', split='train', download=False, transform=trans_f)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=False,
#                                                    drop_last=True)
# unloader = transforms.ToPILImage()
# label_idx = [31, 20, 19, 21, 23, 13]
# for idx, (x, label) in enumerate(train_loader):
#     print(idx, x.size(), label.size())
#     print(label)
#     sup_flag = label[:, 0] != -1
#     print(sup_flag)
#
#     exit()

# print(c)
import torch
img = torch.rand(size=(3, 64, 64))
z = Encoder(img)

position = torch.ceil(z[:8])   # float

img[position]

