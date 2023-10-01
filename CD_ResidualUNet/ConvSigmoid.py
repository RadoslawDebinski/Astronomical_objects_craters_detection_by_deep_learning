import torch.nn as nn


class ConvSigmoid(nn.Module):
    """1x1 Conv Sigmoid"""
    def __init__(self, in_channels, out_channels):
        super(ConvSigmoid, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
