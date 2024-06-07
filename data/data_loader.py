import torch
import torch.utils.data as data
import os
import errno
import math

from torch import FloatTensor, LongTensor
from torch.utils.data.dataset import T_co


class WMTLoader(data.Dataset):
    def __init__(self, transform=None, target_transform=None, split="europarl", pad=True, is_tensor=False):
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.pad = pad
        data = None
        self.data = data
        self.is_tensor = is_tensor

    def __getitem__(self, index) -> T_co:
        source, target = self.data[self.split][index]
        source, target = self.convert_to_tensor(source, target)

        if self.pad:
            return source, target
        else:
            pass

    def convert_to_tensor(self, src, trg):
        """
        Checks if source and target are tensor
        If both are not tensor, they are converted to tensors

        :param src:
        :param trg:
        :return:
        """
        if not torch.is_tensor(src):
            src = FloatTensor([src])
        if not torch.is_tensor(trg):
            trg = LongTensor([trg])
        return src, trg



