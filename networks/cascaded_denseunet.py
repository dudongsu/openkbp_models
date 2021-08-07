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
        self.conv2 = nn.Conv3d(in_channels+mid_channels, out_channels, kernel_size=3, padding=1)
      #  self.BatchNorm3d_1 = nn.BatchNorm3d(mid_channels)
      #  self.BatchNorm3d_2 = nn.BatchNorm3d(out_channels)
        self.BatchNorm3d_1 = nn.InstanceNorm3d(mid_channels, affine=True)
        self.BatchNorm3d_2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.dropout = nn.Dropout(0.8)
        self.act1 = nn.ReLU(inplace=True)

        # initial the weight
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)
        z = torch.cat([x, y], dim=1)
     #   z = self.conv2(y)
     #   z = self.act1(z)
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
            self.up = nn.ConvTranspose3d(in_channels , 64, kernel_size=2, stride=2)
            self.conv = DoubleConv(encode_channels+64, out_channels)


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

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        return self.conv2(x)




class BaseUNet(nn.Module):
    def __init__(self, n_channels, mid_channels, up_channels, bilinear=False):
        super(BaseUNet, self).__init__()
        self.n_channels = n_channels
        self.mid_channels = mid_channels
        self.up_channels = up_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, mid_channels)
        self.down1 = Down(n_channels+mid_channels, mid_channels)
        self.down2 = Down(n_channels+mid_channels*2, mid_channels)
        self.down3 = Down(n_channels+mid_channels*3, mid_channels)
       # factor = 2 if bilinear else 1
        self.down4 = Down(n_channels+mid_channels*4, mid_channels)
        self.up1 = Up(n_channels+mid_channels*5, up_channels, n_channels+mid_channels*5+up_channels, bilinear)
        self.up2 = Up(n_channels+mid_channels*4+mid_channels, up_channels, n_channels+mid_channels*4+up_channels, bilinear)
        self.up3 = Up(n_channels+mid_channels*3+mid_channels, up_channels, n_channels+mid_channels*3+up_channels, bilinear)
        self.up4 = Up(n_channels+mid_channels*2+mid_channels, up_channels, n_channels+mid_channels*2+up_channels, bilinear)
        self.outc = OutConv(n_channels+mid_channels*2+up_channels, 1)

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
    #    dose = self.outc(x)
        return x


class Cascade_DenseUnet(nn.Module):
    def __init__(self, n_channels, mid_channels, up_channels):
        super(Cascade_DenseUnet, self).__init__()

        # list_ch records the number of channels in each stage, eg. [-1, 32, 64, 128, 256, 512]
        self.net_A = BaseUNet(n_channels, mid_channels, up_channels)
        self.net_B = BaseUNet(n_channels+mid_channels*2+up_channels, mid_channels*2, up_channels)

        self.conv_out_A = nn.Conv3d(n_channels+mid_channels*2+up_channels, 1, kernel_size=1, padding=0, bias=True)
        self.conv_out_B = nn.Conv3d(n_channels+mid_channels*2+up_channels+mid_channels*2+up_channels, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)
        out_net_B = self.net_B(out_net_A, dim=1)

        output_A = self.conv_out_A(out_net_A)
        output_B = self.conv_out_B(out_net_B)
        return [output_A, output_B]


class BaseUNet(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(BaseUNet, self).__init__()
        self.encoder = Encoder(in_ch, list_ch)
        self.decoder = Decoder(list_ch)

        # init
        self.initialize()

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

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)

        # Output is a list: [Output]
        return out_decoder


class Cascade_DenseUnet(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B):
        super(Cascade_DenseUnet, self).__init__()

        # list_ch records the number of channels in each stage, eg. [-1, 32, 64, 128, 256, 512]
        self.net_A = BaseUNet(in_ch, list_ch_A)
        self.net_B = BaseUNet(in_ch + list_ch_A[1], list_ch_B)

        self.conv_out_A = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, padding=0, bias=True)
        self.conv_out_B = nn.Conv3d(list_ch_B[1], out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out_net_A = self.net_A(x)
        out_net_B = self.net_B(torch.cat((out_net_A, x), dim=1))

        output_A = self.conv_out_A(out_net_A)
        output_B = self.conv_out_B(out_net_B)
        return [output_A, output_B]