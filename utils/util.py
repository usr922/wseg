import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import random
from functools import partial

class AffineTransformationPool:

    AffineFunc = [
        # partial(F.interpolate, scale_factor=0.3, mode='bilinear', align_corners=True),
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.LinearTransformation,  # color jitter
    ]
    def __init__(self):
        super().__init__()

        pass

    def sample(self):
        fun = random.choice(self.AffineFunc)


