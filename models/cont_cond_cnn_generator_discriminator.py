'''
https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

chainer: https://github.com/pfnet-research/sngan_projection
'''

# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F

# from spectral_normalization import SpectralNorm
import numpy as np
from torch.nn.utils import spectral_norm



channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=bias) #h=h
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=bias)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if stride != 1:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=256
DISC_SIZE=256

class cont_cond_cnn_generator(nn.Module):
    def __init__(self, nz=128):
        super(cont_cond_cnn_generator, self).__init__()
        self.z_dim = nz

        # self.dense = nn.Sequential(
        #     nn.Linear(self.z_dim, 4 * 4 * (GEN_SIZE*16), bias=True),
        #     # nn.BatchNorm1d(4 * 4 * (GEN_SIZE*16)),
        #     nn.ReLU()
        # )
        self.dense = nn.Linear(self.z_dim, 4 * 4 * (GEN_SIZE*16), bias=True)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.genblock1 = ResBlockGenerator((GEN_SIZE*16), (GEN_SIZE*8), bias=True) #4--->8
        self.genblock2 = ResBlockGenerator((GEN_SIZE*8), (GEN_SIZE*4), bias=True) #8--->16
        self.genblock3 = nn.Sequential(
            ResBlockGenerator((GEN_SIZE*4), GEN_SIZE*2, bias=True), #16--->32
            ResBlockGenerator(GEN_SIZE*2, GEN_SIZE, bias=True), #32--->64
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z, y):
        y = y.view(-1,1)

        z = z.view(z.size(0), z.size(1))
        print("kyoma z.shape")
        print(z.shape)
        print("kyoma y.shape")
        print(y.shape)
        tmp1 = self.dense(z)
        print("kyoma tmp = self.dense(z).shape")
        print(tmp1.shape)
        tmp2 = y.repeat(1, 4 * 4 * (GEN_SIZE*16))
        print("kyoma tmp2 = y.repeat(1, 4 * 4 * (GEN_SIZE*16)).shape")
        print(tmp2.shape)
        out = tmp1 + tmp2
        out = out.view(-1, (GEN_SIZE*16), 4, 4)

        out = self.genblock1(out) #+ y.view(-1, 1).repeat(1,GEN_SIZE*16*8*8).view(-1, GEN_SIZE*16, 8, 8)
        out = self.genblock2(out) #+ y.view(-1, 1).repeat(1,GEN_SIZE*4*16*16).view(-1, GEN_SIZE*4, 16, 16)
        out = self.genblock3(out)

        return out


class cont_cond_cnn_discriminator(nn.Module):
    def __init__(self):
        super(cont_cond_cnn_discriminator, self).__init__()

        self.discblock1 = nn.Sequential(
            FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2, bias=True), #256--->128, channle 3-->256
            ResBlockDiscriminator(DISC_SIZE  , DISC_SIZE*2, stride=2, bias=True), #128--->64, channle 256-->512
            ResBlockDiscriminator(DISC_SIZE*2  , DISC_SIZE*4, stride=4, bias=True), #64--->16, channle 512-->1024
        )
        self.discblock2 = ResBlockDiscriminator(DISC_SIZE*4, DISC_SIZE*8, stride=2, bias=True) #16->8, 1024-->2048
        self.discblock3 = nn.Sequential(
            ResBlockDiscriminator(DISC_SIZE*8, DISC_SIZE*16, stride=2, bias=True), #8--->4; 2048-->4096
            nn.ReLU(),
        )

        self.linear1 = nn.Linear(DISC_SIZE*16*4*4, 1, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        self.linear1 = spectral_norm(self.linear1)
        self.linear2 = nn.Linear(1, DISC_SIZE*16*4*4, bias=False)
        nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        self.linear2 = spectral_norm(self.linear2)
        self.sigmoid = nn.Sigmoid()

        # self.linear1 = nn.Linear(DISC_SIZE*16, 1, bias=True)
        # nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        # self.linear1 = spectral_norm(self.linear1)
        # self.linear2 = nn.Linear(1, DISC_SIZE*16, bias=False)
        # nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        # self.linear2 = spectral_norm(self.linear2)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x[B, 3, 256, 256], y[B]
        y = y.view(-1,1)
        # y[B,1]

        output = self.discblock1(x)
        output = self.discblock2(output)
        output = self.discblock3(output)

        output = output.view(-1, DISC_SIZE*16*4*4)
        output_y = torch.sum(output*self.linear2(y+1), 1, keepdims=True)
        output = self.sigmoid(self.linear1(output) + output_y)

        # output = torch.sum(output, dim=(2,3))
        # output_y = torch.sum(output*self.linear2(y+1), 1, keepdims=True)
        # output = self.sigmoid(self.linear1(output) + output_y)

        return output.view(-1, 1)
