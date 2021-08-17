import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import *


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
    #    self.BatchNorm3d_1 = nn.BatchNorm3d(mid_channels)
    #    self.BatchNorm3d_2 = nn.BatchNorm3d(out_channels)
        self.BatchNorm3d_1 = nn.InstanceNorm3d(mid_channels, affine=True)
        self.BatchNorm3d_2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.dropout = nn.Dropout(0.5)
        self.act1 = nn.ReLU(inplace=True)

        # initial the weight
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        y = self.conv1(x)
    #    y = self.BatchNorm3d_1(y)
        y = self.act1(y)
    #    m = nn.LayerNorm(y.size()[1:])
     #   y = self.dropout(y)
        z = self.conv2(y)
    #    z = self.BatchNorm3d_2(z)
        z = self.act1(z)
     #   z = nn.LayerNorm(z.size()[2:])(z)
     #   z = torch.cat([y, z], dim=1)
        return z




class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, encode_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(encode_channels+in_channels//2, out_channels)


    def forward(self, x1, encode):
        x1 = self.up(x1)
        x = torch.cat([encode, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(2, out_channels, kernel_size=1)
        self.act1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        return self.conv2(x)


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
       # factor = 2 if bilinear else 1
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256, 256, bilinear)
        self.up2 = Up(256, 128, 128, bilinear)
        self.up3 = Up(128, 64, 64, bilinear)
        self.up4 = Up(64, 32, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    @staticmethod
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        dose = self.outc(x)
        return dose