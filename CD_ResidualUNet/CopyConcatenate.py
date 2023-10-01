import torch


def CopyConcatenate(x1, x2):
    return torch.cat([x1, x2], dim=1)
