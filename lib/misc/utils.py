import torch
import torch.nn as nn

def get_params(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            for p in m.parameters():
                if p.requires_grad:
                    yield p
        elif isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                if p.requires_grad:
                    yield p
