import torch.nn as nn

from CD_ResidualUNet.ConvBNReLU import ConvBNReLU
from CD_ResidualUNet.ConvSigmoid import ConvSigmoid
from CD_ResidualUNet.ConvTranspose import ConvTranspose
from CD_ResidualUNet.CopyConcatenate import CopyConcatenate
from CD_ResidualUNet.DropoutConv import DropoutConv
from CD_ResidualUNet.MaxPooling import MaxPooling
from CD_ResidualUNet.ResidualConv import ResidualConv


class ResidualUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResidualUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Block 1
        self.convBNReLU1 = ConvBNReLU(n_channels, 112)
        self.residualConv1 = ResidualConv(112, 112)

        # Between block 1 and 2
        self.maxPooling12 = MaxPooling()

        # Block 2
        self.convBNReLU2 = ConvBNReLU(112, 224)
        self.residualConv2 = ResidualConv(224, 224)

        # Between block 2 and 3
        self.maxPooling23 = MaxPooling()

        # Block 3
        self.convBNReLU3 = ConvBNReLU(224, 448)
        self.residualConv3 = ResidualConv(448, 448)

        # Between block 3 and 4
        self.maxPooling34 = MaxPooling()

        # Block 4
        self.convBNReLU4 = ConvBNReLU(448, 448)
        self.residualConv4 = ResidualConv(448, 448)

        # Between block 4 and 5
        self.convTranspose45 = ConvTranspose(448, 448)

        # Block 5
        self.dropoutConv5 = DropoutConv(896, 448)
        self.residualConv5 = ResidualConv(448, 448)

        # Between block 5 and 6
        self.convTranspose56 = ConvTranspose(448, 224)

        # Block 6
        self.dropoutConv6 = DropoutConv(448, 224)
        self.residualConv6 = ResidualConv(224, 224)

        # Between block 6 and 7
        self.convTranspose67 = ConvTranspose(224, 112)

        # Block 7
        self.dropoutConv7 = DropoutConv(224, 112)
        self.residualConv7 = ResidualConv(112, 112)
        self.convSigmoid7 = ConvSigmoid(112, n_classes)

    def forward(self, x):
        # Block 1
        x1 = self.convBNReLU1(x)
        x1 = self.residualConv1(x1)

        # Between block 1 and 2
        x2 = self.maxPooling12(x1)

        # Block 2
        x2 = self.convBNReLU2(x2)
        x2 = self.residualConv2(x2)

        # Between block 2 and 3
        x3 = self.maxPooling23(x2)

        # Block 3
        x3 = self.convBNReLU3(x3)
        x3 = self.residualConv3(x3)

        # Between block 3 and 4
        x4 = self.maxPooling34(x3)

        # Block 4
        x4 = self.convBNReLU4(x4)
        x4 = self.residualConv4(x4)

        # Between block 4 and 5
        x5 = self.convTranspose45(x4)

        # Block 5
        x5 = CopyConcatenate(x3, x5)
        x5 = self.dropoutConv5(x5)
        x5 = self.residualConv5(x5)

        # Between block 5 and 6
        x6 = self.convTranspose56(x5)

        # Block 6
        x6 = CopyConcatenate(x2, x6)
        x6 = self.dropoutConv6(x6)
        x6 = self.residualConv6(x6)

        # Between block 6 and 7
        x7 = self.convTranspose67(x6)

        # Block 7
        x7 = CopyConcatenate(x1, x7)
        x7 = self.dropoutConv7(x7)
        x7 = self.residualConv7(x7)
        x7 = self.convSigmoid7(x7)

        return x7
