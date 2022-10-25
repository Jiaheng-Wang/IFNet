import torch
import random


class RandomCrop:
    def __init__(self, points = 750):
        self.points = points

    def __call__(self, x: torch.tensor, *args):
        N, C, T = x.shape
        s = random.randint(0, T - self.points)
        return x[..., s:s + self.points]