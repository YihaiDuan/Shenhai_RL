"""
@Time ： 2021/5/18 10:21
@Auth ： Duan Yihai
@File ：rl_dataset.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""

import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from functools import partial
import pandas
import PIL
import matplotlib.pyplot as plt
import numpy as np

class RLDataset(Dataset):
    def __init__(self, root_dir, label_file='attr', transform=None):
        self.label_file = label_file
        self.root_dir = root_dir
        self.transform = transform
        self.img_names = pandas.read_csv(os.path.join(self.root_dir, 'partition.txt'), header=None, delim_whitespace=True)
        attr = pandas.read_csv(os.path.join(self.root_dir, "list_attr.txt"), delim_whitespace=True, header=1)
        self.attr = torch.as_tensor(attr.values)

    def __getitem__(self, index):

        X = PIL.Image.open(os.path.join(self.root_dir, "img", self.img_names.iloc[index, 0]))

        target = self.attr[index, :]

        if self.transform is not None:
            X = self.transform(X)

        return X, target

    def __len__(self):
        return len(self.attr)
