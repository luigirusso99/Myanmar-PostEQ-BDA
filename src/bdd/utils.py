import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_bn(m):
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def initialize_model(m):
    init_kaiming(m)
    init_xavier(m)
    init_bn(m)
