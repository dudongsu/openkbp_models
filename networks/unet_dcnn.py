import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_2 = nn.Sequential(
            nn.MaxPool3d(2),
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)
        )
        self.encoder_3 = nn.Sequential(
            nn.MaxPool3d(2),
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)

        return [out_encoder_1, out_encoder_2, out_encoder_3] 


class DenseFeaureAggregation(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch):
        super(DenseFeaureAggregation, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, dilation=2, kernel_size=3, padding=2, stride=1, bias=True),
            nn.ReLU(inplace=True),

        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch + base_ch, base_ch, dilation=3, kernel_size=3, padding=3, stride=1, bias=True),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_ch + 2 * base_ch, base_ch, dilation=5, kernel_size=3, padding=5, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_ch + 3 * base_ch, base_ch, dilation=7, kernel_size=3, padding=7, stride=1, bias=True),
            nn.ReLU(inplace=True),

        )
        self.conv5 = nn.Sequential( 
            nn.Conv3d(in_ch + 4 * base_ch, base_ch, dilation=9, kernel_size=3, padding=9, stride=1, bias=True),
            nn.ReLU(inplace=True),

        )

        self.conv_out = nn.Sequential(
            nn.Conv3d(in_ch + 5 * base_ch, out_ch, dilation=1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3 = out_encoder
        out_ = self.conv1(out_encoder_3)
        concat_ = torch.cat((out_, out_encoder_3), dim=1)
        out_ = self.conv2(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv3(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv4(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_ = self.conv5(concat_)
        concat_ = torch.cat((concat_, out_), dim=1)
        out_dcnn = self.conv_out(concat_)
        return out_dcnn



class Decoder(nn.Module):
    def __init__(self, list_ch):
        super(Decoder, self).__init__()

        self.upconv_2 = UpConv(list_ch[4], list_ch[2])
        self.decoder_conv_2 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1)
        )

        self.upconv_1 = UpConv(list_ch[2], list_ch[1])
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )

    def forward(self, out_encoder, out_dcnn):
        out_encoder_1, out_encoder_2, out_encoder_3 = out_encoder
      #  print ('out_cnn size', out_dcnn.size())
      #  print ('out_encoder_2 size', out_encoder_2.size())

        out_decoder_2 = self.decoder_conv_2(
            torch.cat((self.upconv_2(out_dcnn), out_encoder_2), dim=1)
        )
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((self.upconv_1(out_decoder_2), out_encoder_1), dim=1)
        )

        return out_decoder_1


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



class UNet_dcnn(nn.Module):
    def __init__(self, in_ch=11, list_ch=[-1, 16, 32, 64, 64]):
        super(UNet_dcnn, self).__init__()
        self.encoder = Encoder(in_ch, list_ch)
        self.decoder = Decoder(list_ch)
        self.dcnn = DenseFeaureAggregation(list_ch[3], list_ch[4], 16)
        self.out = OutConv(list_ch[1],1)
        # init
#        self.initialize()

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
        print('# random init dcnn weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.dcnn.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_dcnn =  self.dcnn(out_encoder)
        out_decoder = self.decoder(out_encoder, out_dcnn)
        output = self.out(out_decoder)
        # Output is a list: [Output]
        return output


'''
class Model(nn.Module):
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B):
        super(Model, self).__init__()

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

'''