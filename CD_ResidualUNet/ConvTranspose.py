import torch.nn as nn


class ConvTranspose(nn.Module):
    """3x3 ConvTranspose"""
    def __init__(self, in_channels, out_channels):
        super(ConvTranspose, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        return self.conv_transpose(x)
