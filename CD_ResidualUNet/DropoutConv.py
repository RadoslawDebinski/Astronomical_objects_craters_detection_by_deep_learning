import torch.nn as nn


class DropoutConv(nn.Module):
    """Dropout + 3x3 Conv"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super(DropoutConv, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        return x
