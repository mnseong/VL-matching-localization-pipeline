import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import importlib
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225)),
                                ])

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2)
        self.gn = nn.GroupNorm(num_groups=32,num_channels=num_out_layers) # Group Norm with 32

    def forward(self, x):
        return F.elu(self.gn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale  = scale
        self.conv   = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


class upconv_like(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(upconv_like, self).__init__()

        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x, target):
        x = F.upsample(x,size=target.shape[2:],mode='bilinear')
        return self.conv(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PositionwiseNorm2(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon    = epsilon
        self.conv1      = nn.Conv2d(1,1,kernel_size=7,stride=1,padding=3)
        self.conv2      = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
    def forward(self,x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(self.epsilon).sqrt()
        output = (x - mean) / std
        map = torch.mean(x,dim=1, keepdim=True)
        map1 = self.conv1(map)
        map2 = self.conv2(map)
        return output*map1 + map2


class Adaffusion(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels   = num_channels
        self.avg_pool       = nn.AdaptiveAvgPool2d(1)
        self.fc1            = nn.Linear(128,64)
        self.relu1          = nn.ReLU()
        self.fc2            = nn.Linear(64,128)
        self.fc3            = nn.Linear(128, 64)
        self.relu2          = nn.ReLU()
        self.fc4            = nn.Linear(64, 128)

    def forward(self, result, x):
        avg_out1 = self.fc2(self.relu1(self.fc1(self.avg_pool(x).squeeze(-1).squeeze(-1)))).unsqueeze(-1).unsqueeze(-1)
        avg_out2 = self.fc4(self.relu2(self.fc3(self.avg_pool(x).squeeze(-1).squeeze(-1)))).unsqueeze(-1).unsqueeze(-1)
        return result * avg_out1 + avg_out2


class ChannelwiseNorm(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5, affusion=False, track_running_stats=False):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        if affusion:
            self.affusion = Adaffusion(num_features)
        else:
            self.affusion = None

        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape

        if self.training or not self.track_running_stats:
            # All dims except for B and C
            mu = x.mean(dim=(2, 3))
            sigma = x.var(dim=(2, 3), unbiased=False)
        else:
            mu, sigma = self.running_mean, self.running_var
            b = 1

        if self.training and self.track_running_stats:
            sigma_unbiased = sigma * ((h * w) / ((h * w) - 1))
            self.running_mean   = self.running_mean * (1 - self.momentum) + mu.mean(dim=0) * self.momentum
            self.running_var    = self.running_var * (1 - self.momentum) + sigma_unbiased.mean(dim=0) * self.momentum

        mu = mu.reshape(b, c, 1, 1)
        sigma = sigma.reshape(b, c, 1, 1)
        result = (x - mu) / torch.sqrt(sigma + self.eps)

        if self.affusion is not None:
            result = self.affusion(result)

        return result


import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

# Test the implementation
if __name__ == "__main__":
    # Assuming input channel is 64
    input_tensor = torch.randn(1, 128, 264, 400)
    cbam_module = CBAM(in_channels=128)
    output = cbam_module(input_tensor)
    print("Output shape:", output.shape)


class ResUNet_F2R(nn.Module):
    """
    F2R-Backbone: Feature-Fusion-ResUNet Backbone
    """
    def __init__(self,
                 encoder='resnet152',
                 pretrained=True,
                 fusion_out_ch=128
                 ):

        super(ResUNet_F2R, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        self.firstconv      = resnet.conv1  # H/2
        self.firstbn        = resnet.bn1
        self.firstrelu      = resnet.relu
        self.firstmaxpool   = resnet.maxpool  # H/4

        # Encoder
        self.layer1 = resnet.layer1  # H/4
        self.layer2 = resnet.layer2  # H/8
        self.layer3 = resnet.layer3  # H/16

        # Decoder
        self.upconv3    = upconv(filters[2], 512, 3, 2)
        self.iconv3     = conv(filters[1] + 512, 512, 3, 1)
        self.upconv2    = upconv(512, 256, 3, 2)
        self.iconv2     = conv(filters[0] + 256, 256, 3, 1)
        self.upconv1    = upconv(256, 256, 3, 2)
        self.iconv1     = conv(256+64, 256, 3, 1)

        # Feature Fusion Block
        self.side3          = upconv_like(1024, fusion_out_ch, 3)
        self.side2          = upconv_like(512, fusion_out_ch, 3)
        self.side1          = conv(256, fusion_out_ch, 1, 1)
        self.fusion_conv    = nn.Conv2d(3*fusion_out_ch,fusion_out_ch,1)

        # Cross Norm. Layer
        self.fusion_pn = PositionwiseNorm2()
        self.fusion_cn = ChannelwiseNorm(fusion_out_ch)
        self.fuse_weight_fusion_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_fusion_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_fusion_1.data.fill_(0.7)
        self.fuse_weight_fusion_2.data.fill_(0.3)

        self.out_channels = 192

        self.f_upconv    = upconv(128, 128, 3, 2)
        self.f_conv = conv(128, 64, 3, 1)
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def name(self):
        return 'ResUNet_F2R'

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):

        # Encoder
        x       = self.firstrelu(self.firstbn(self.firstconv(x)))
        x_store = x
        x_first = self.firstmaxpool(x)

        x1 = self.layer1(x_first)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        # Decoder
        x   = self.upconv3(x3)
        x   = self.skipconnect(x2, x)
        x2d = self.iconv3(x)

        x = self.upconv2(x2d)
        x = self.skipconnect(x1, x)
        x1d = self.iconv2(x)

        x = self.upconv1(x1d)
        x = self.skipconnect(x_store, x)
        # print(x.shape)
        x = self.iconv1(x)
        # print(x.shape)

        # Feature Fusion output
        d1          = self.side1(x)
        d2          = self.side2(x2d, d1)
        d3          = self.side3(x3, d1)
        x_fusion    = self.fusion_conv(torch.cat((d1,d2,d3),1))  # H/4

        x_fusion = self.f_upconv(x_fusion)

        del x, x2, d1, d2, d3, x2d

        # # Shared Coupling-bridge Normalization
        desc1       = self.fusion_pn(x_fusion)
        desc2       = self.fusion_cn(x_fusion)
        x_fusion_cn = desc1 * (self.fuse_weight_fusion_1/(self.fuse_weight_fusion_1+self.fuse_weight_fusion_2)) + \
                      desc2 * (self.fuse_weight_fusion_2/(self.fuse_weight_fusion_1+self.fuse_weight_fusion_2))

        # print(x_fusion_cn.shape)

        # final = self.f_upconv(x_fusion_cn)
        # print(final.shape)
        final = self.f_conv(x_fusion_cn)
        # print(final.shape)
        final = self.fc(final)
        # print(final.shape)

        # return {
        #     'global_map':       x3,             # Coarse level
        #     'desc_map':         x_fusion,       # D_desc
        #     'local_map':        x_fusion_cn,    # F_cn
        #     'local_map_first':  x_first         # Initial feature maps
        # }

        return final




import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        """ Residual block과 channel size를 맞추기 위한 conv operation """
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
        )

    def forward(self, inputs):
        r = self.conv_block(inputs)
        s = self.shortcut(inputs)

        skip = r + s
        return skip

class Dacon_ResUNet(nn.Module):
    def __init__(self, num_classes):
        super(Dacon_ResUNet, self).__init__()
        self.num_classes = num_classes

        """ Encoder input layer """
        self.contl_1 = self.input_block(in_channels=3, out_channels=64)
        self.contl_2 = self.input_skip(in_channels=3, out_channels=64)

        """ Residual encoder block """
        self.resdl_1 = ResidualBlock(64, 128, 2, 1)
        self.resdl_2 = ResidualBlock(128, 256, 2, 1)
        #self.resdl_3 = ResidualBlock(256, 512, 2, 1)

        """ Encoder decoder skip connection """
        self.middle = ResidualBlock(256, 512, 2, 1)

        """ Decoder block """
        self.expnl_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                          kernel_size=2, stride=2, padding=0)
        self.expnl_1_cv = ResidualBlock(256+256, 256, 1, 1)
        self.expnl_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=2, stride=2, padding=0)
        self.expnl_2_cv = ResidualBlock(128+128, 128, 1, 1)
        self.expnl_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=2, stride=2, padding=0)
        self.expnl_3_cv = ResidualBlock(64+64, 64, 1, 1)
        # self.expnl_4 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
        #                                   kernel_size=2, stride=2, padding=0)
        # self.expnl_4_cv = ResidualBlock(128+64, 64, 1, 1)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def input_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=out_channels),
                              nn.ReLU(),
                              nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                              )
        return block

    def input_skip(self, in_channels, out_channels):
        skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        return skip

    def forward(self, X):
        contl_1_out = self.contl_1(X) # 64
        contl_2_out = self.contl_2(X) # 64
        input_out = contl_1_out + contl_2_out

        resdl_1_out = self.resdl_1(input_out) # 128
        resdl_2_out = self.resdl_2(resdl_1_out) # 256
        #resdl_3_out = self.resdl_3(resdl_2_out) # 512

        middle_out = self.middle(resdl_2_out) # 512

        expnl_1_out = self.expnl_1(middle_out)
        expnl_1_cv_out = self.expnl_1_cv(torch.cat((expnl_1_out, resdl_2_out), dim=1)) # 512
        expnl_2_out = self.expnl_2(expnl_1_cv_out) # 256
        expnl_2_cv_out = self.expnl_2_cv(torch.cat((expnl_2_out, resdl_1_out), dim=1))
        expnl_3_out = self.expnl_3(expnl_2_cv_out)
        expnl_3_cv_out = self.expnl_3_cv(torch.cat((expnl_3_out, contl_1_out), dim=1))
        # expnl_4_out = self.expnl_4(expnl_3_cv_out)
        # expnl_4_cv_out = self.expnl_4_cv(torch.cat((expnl_4_out, input_out), dim=1))

        out = self.output(expnl_3_cv_out)
        return out