import torch.nn as nn


class MaxPooling(nn.Module):
    """2x2 MaxPooling"""
    def __init__(self):
        super(MaxPooling, self).__init__()
        self.maxPooling = nn.MaxPool2d(kernel_size=2, stride=2)     # stride?

    def forward(self, x):
        x = self.maxPooling(x)
        return x
