import torch
import torch.nn as nn


class ResidualConv(nn.Module):
    """Input -> 3x3 Conv -> BN+ReLU -> 3x3 Conv -> +Input -> BN+ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        input_x = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = torch.add(x, input_x)

        x = self.bn2(x)
        x = self.relu2(x)

        return x
