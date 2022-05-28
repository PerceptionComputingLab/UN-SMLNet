import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

'''
class OutConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, upsample='trilinear'):
        super(OutConvBlock, self).__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if upsample == 'transpose':
            ops.append(nn.ConvTranspose3d(n_filters_out, n_filters_out, stride, padding=0, stride=stride))
        elif upsample == 'trilinear':
            ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
            ops.append(nn.Conv3d(n_filters_out, n_filters_out, kernel_size=3, padding=1))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
'''
class OutConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, upsample='trilinear'):
        super(OutConvBlock, self).__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=1, padding=0))

        if upsample == 'transpose':
            temp = stride//2
            while temp != 1:
                #每次上采样步长为2
                ops.append(nn.ConvTranspose3d(n_filters_out, n_filters_out, kernel_size=1, stride=2, padding=0))
                temp //= 2
            ops.append(nn.ConvTranspose3d(n_filters_out, n_filters_out, kernel_size=1, stride=2, padding=0))
            ops.append(nn.Conv3d(n_filters_out, n_filters_out, kernel_size=1, padding=0))
        elif upsample == 'trilinear':
            temp = stride//2
            while temp != 1:
                # 每次上采样步长为2
                ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
                temp //= 2
            ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
            ops.append(nn.Conv3d(n_filters_out, n_filters_out, kernel_size=1, padding=0))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class OutConvBlockfor1(nn.Module):
    def __init__(self, n_filters_in, n_filters_out):
        super(OutConvBlockfor1, self).__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=1, stride=1, padding=0))
        ops.append(nn.Conv3d(n_filters_out, n_filters_out, kernel_size=1, stride=1, padding=0))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
class ConvAttentionBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, ratio):
        super(ConvAttentionBlock, self).__init__()

        n_filters_cut = n_filters_in // ratio
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_cut, 3, padding=1))
        ops.append(nn.BatchNorm3d(n_filters_cut))
        ops.append(nn.ReLU(inplace=True))
        ops.append(nn.Conv3d(n_filters_cut, n_filters_out, 1, padding=0))
        ops.append(nn.Sigmoid())
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        y = self.conv(x)
        # res add
        return x * y + x

class VNetMultiHead(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', ratio=8, has_att=False, has_enout=True,has_dropout=False):
        super(VNetMultiHead, self).__init__()
        self.has_dropout = has_dropout
        self.has_att = has_att
        self.has_enout = has_enout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)
        # auxiliary prediction, before downsampling
        self.block_one_cab = ConvAttentionBlock(n_filters, n_filters, ratio=ratio)
        self.block_one_out = OutConvBlockfor1(n_filters,n_classes)#OutConvBlock(n_filters, n_classes, stride=1, upsample='transpose')#

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)
        # auxiliary prediction, before downsampling
        self.block_two_cab = ConvAttentionBlock(n_filters * 2, n_filters * 2, ratio=ratio)
        self.block_two_out = OutConvBlock(n_filters * 2, n_classes, stride=2, upsample='trilinear')

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)
        # auxiliary prediction, before downsampling
        self.block_three_cab = ConvAttentionBlock(n_filters * 4, n_filters * 4, ratio=ratio)
        self.block_three_out = OutConvBlock(n_filters * 4, n_classes, stride=4, upsample='trilinear')

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)
        # auxiliary prediction, before downsampling
        self.block_four_cab = ConvAttentionBlock(n_filters * 8, n_filters * 8, ratio=ratio)
        self.block_four_out = OutConvBlock(n_filters * 8, n_classes, stride=8, upsample='trilinear')


        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)
        # auxiliary prediction, before upsampling
        self.block_six_cab = ConvAttentionBlock(n_filters * 8, n_filters * 8, ratio=ratio)
        self.block_six_out = OutConvBlock(n_filters * 8, n_classes, stride=8, upsample='trilinear')


        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)
        # auxiliary prediction, before upsampling
        self.block_seven_cab = ConvAttentionBlock(n_filters * 4, n_filters * 4, ratio=ratio)
        self.block_seven_out = OutConvBlock(n_filters * 4, n_classes, stride=4, upsample='trilinear')

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)
        # auxiliary prediction, before upsampling
        self.block_eight_cab = ConvAttentionBlock(n_filters * 2, n_filters * 2, ratio=ratio)
        self.block_eight_out = OutConvBlock(n_filters * 2, n_classes, stride=2, upsample='trilinear')

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_cab = ConvAttentionBlock(n_filters, n_filters, ratio=ratio)
        self.block_nine_out =OutConvBlockfor1(n_filters,n_classes)#OutConvBlock(n_filters, n_classes, stride=1, upsample='transpose')#

        self.logits_out = nn.Conv3d(n_classes*4, n_classes, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):

        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)
        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        # 输出
        if self.has_att:
            x6 = self.block_six_cab(x6)
            x7 = self.block_seven_cab(x7)
            x8 = self.block_eight_cab(x8)
            x9 = self.block_nine_cab(x9)
        x6_out = self.block_six_out(x6)
        x7_out = self.block_seven_out(x7)
        x8_out = self.block_eight_out(x8)
        x9_out = self.block_nine_out(x9)

        decoder_out = torch.cat((x6_out, x7_out, x8_out, x9_out), 1)
        decoder_out_logits = self.logits_out(decoder_out)
        #out_logits = self.logits_out(x9)
        #out_dis = self.dis_out(x9)

        return decoder_out_logits
    
    def encoder_out(self, features):
        # 输出
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        if self.has_att:
            x1 = self.block_one_cab(x1)
            x2 = self.block_two_cab(x2)
            x3 = self.block_three_cab(x3)
            x4 = self.block_four_cab(x4)
        x1_out = self.block_one_out(x1)
        x2_out = self.block_two_out(x2)
        x3_out = self.block_three_out(x3)
        x4_out = self.block_four_out(x4)
        encoder_out = torch.cat((x1_out, x2_out, x3_out, x4_out), 1)
        encoder_out_logits = self.logits_out(encoder_out)
        return encoder_out_logits


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)

        decoder_out_logits = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        if self.has_enout:
            encoder_out_logits = self.encoder_out(features)
            return decoder_out_logits, encoder_out_logits
        else:
            return decoder_out_logits

