import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double convolution module representing application of:
        - 2 × (Conv 3×3 → Batch Normalization → ReLU)
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    """
    Encoder module representing application of:
        - Double convolution 2 × (Conv 3×3 → Batch Normalization → ReLU)
        - Max-pooling 2×2
    """

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)

        return x, p


class Decoder(nn.Module):
    """
    Decoder module representing application of:
        - Transposed Conv 2×2
        - Concatenation
        - Double convolution 2 × (Conv 3×3 → Batch Normalization → ReLU)
    """

    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, e):
        x = self.up(x)
        x = torch.cat([e, x], dim=1)
        x = self.conv(x)

        return x


class AttentionGate(nn.Module):
    """
    Attention gate module representing application of attention mechanism
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.resample_down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.resample_up = nn.Upsample(scale_factor=2)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1)
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g = self.W_g(g)
        x = self.W_x(x)
        x_d = self.resample_down(x)

        sum_relu = self.relu(g + x_d)
        psi = self.psi(sum_relu)
        psi = self.sigmoid(psi)

        alpha = self.resample_up(psi)

        return x * alpha


class FinalConv(nn.Module):
    """
    Final convolution module representing application of:
        - Conv 1×1 → Batch Normalization
    """

    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.final_conv(x)


class AttentionUNet(nn.Module):
    """
    Complete attention U-Net architecture module
    """

    def __init__(self, in_channels=1, out_channels=1, n_filter=64):
        super(AttentionUNet, self).__init__()

        n_filters = [n_filter, n_filter * 2, n_filter * 4, n_filter * 8]

        self.encoder1 = Encoder(in_channels, n_filters[0])
        self.encoder2 = Encoder(n_filters[0], n_filters[1])
        self.encoder3 = Encoder(n_filters[1], n_filters[2])

        self.bridge = DoubleConv(n_filters[2], n_filters[3])

        self.attention_block1 = AttentionGate(n_filters[3], n_filters[2], n_filters[3] // 2)
        self.attention_block2 = AttentionGate(n_filters[2], n_filters[1], n_filters[2] // 2)
        self.attention_block3 = AttentionGate(n_filters[1], n_filters[0], n_filters[1] // 2)

        self.decoder1 = Decoder(n_filters[3], n_filters[2])
        self.decoder2 = Decoder(n_filters[2], n_filters[1])
        self.decoder3 = Decoder(n_filters[1], n_filters[0])

        self.final_conv = FinalConv(n_filters[0], out_channels)

    def forward(self, x):
        e1, x = self.encoder1(x)
        e2, x = self.encoder2(x)
        e3, x = self.encoder3(x)

        b = self.bridge(x)

        d1 = self.decoder1(b, self.attention_block1(b, e3))
        d2 = self.decoder2(d1, self.attention_block2(d1, e2))
        d3 = self.decoder3(d2, self.attention_block3(d2, e1))

        output = self.final_conv(d3)

        return output
