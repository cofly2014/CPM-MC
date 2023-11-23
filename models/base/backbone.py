#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Backbone/Meta architectures. """

import torch
import torch.nn as nn
import torchvision
from util.registry import Registry
from models.base.base_blocks import (
    Base3DResStage, STEM_REGISTRY, BRANCH_REGISTRY, InceptionBaseConv3D
)


BACKBONE_REGISTRY = Registry("Backbone")

_n_conv_resnet = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}

@BACKBONE_REGISTRY.register()
class Identity(nn.Module):
    def __init__(self, cfg):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
