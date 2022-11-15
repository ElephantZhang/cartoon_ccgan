import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# PyTorch implementation by vinesmsuic
# Referenced from official tensorflow implementation: https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/train_code/network.py
# slim.convolution2d uses constant padding (zeros).

dim_embed = 64

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False, device=config.DEVICE)
        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False, device=config.DEVICE)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False, device=config.DEVICE)

    def forward(self, x, y0, y1):
        out = self.bn(x)
        gamma0 = self.embed_gamma(y0).view(-1, self.num_features, 1, 1)
        beta0 = self.embed_beta(y0).view(-1, self.num_features, 1, 1)
        gamma1 = self.embed_gamma(y1).view(-1, self.num_features, 1, 1)
        beta1 = self.embed_beta(y1).view(-1, self.num_features, 1, 1)
        out = out + gamma0*out + beta0 + gamma1*out + beta1
        return out


class ResidualBlock(nn.Module): # B, 128, 64, 64
    def __init__(self, channels, kernel_size, stride, padding, padding_mode):
        super().__init__()
        # self.block = nn.Sequential(
        #     ConditionalBatchNorm2d(channels, dim_embed),
        #     nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     ConditionalBatchNorm2d(channels, dim_embed),
        #     nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
        # )
        self.cond_batchnorm = ConditionalBatchNorm2d(channels, dim_embed)
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, device=config.DEVICE)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.cond_batchnrom = ConditionalBatchNorm2d(channels, dim_embed)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, device=config.DEVICE)

    def forward(self, x, y_surface, y_texture):
        #Elementwise Sum (ES)
        x_ = self.cond_batchnorm(x, y_surface, y_texture)
        x_ = self.conv0(x_)
        x_ = self.relu(x_)
        x_ = self.cond_batchnorm(x_, y_surface, y_texture)
        x_ = self.conv1(x_)
        # return x + self.block(x)
        return x + x_

class Up1(nn.Module):
    def __init__(self, num_features, kernel_size, stride, padding, padding_mode):
        super().__init__()
        # self.up1 = nn.Sequential(
        #     #k3n128s1 (should be k3n64s1?)
        #     nn.Conv2d(num_features*4, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
        #     ConditionalBatchNorm2d(num_features*2, dim_embed),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        # )
        self.conv = nn.Conv2d(num_features*4, num_features*2, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, device=config.DEVICE)
        self.cond_batchnorm = ConditionalBatchNorm2d(num_features*2, dim_embed)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x, y_surface, y_texture):
        x = self.conv(x)
        x = self.cond_batchnorm(x, y_surface, y_texture)
        x = self.relu(x)
        return x

class Up2(nn.Module):
    # self.up2 = nn.Sequential(
    #         #k3n64s1
    #         nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
    #         ConditionalBatchNorm2d(num_features*2, dim_embed),
    #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #         #k3n64s1 (should be k3n32s1?)
    #         nn.Conv2d(num_features*2, num_features, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
    #         ConditionalBatchNorm2d(num_features, dim_embed),
    #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #     )
    def __init__(self, num_features, kernel_size, stride, padding, padding_mode):
        super().__init__()
        self.conv0 = nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, device=config.DEVICE)
        self.cond_batchnorm0 = ConditionalBatchNorm2d(num_features*2, dim_embed)
        self.relu0 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(num_features*2, num_features, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, device=config.DEVICE)
        self.cond_batchnorm1 = ConditionalBatchNorm2d(num_features, dim_embed)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x, y_surface, y_texture):
        x = self.conv0(x)
        x = self.cond_batchnorm0(x, y_surface, y_texture)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.cond_batchnorm1(x, y_surface, y_texture)
        x = self.relu1(x)
        return x

class Last(nn.Module):
    def __init__(self, num_features, out_channels, kernel_size, stride, padding, padding_mode):
        # self.last = nn.Sequential(
        #     #k3n32s1
        #     ConditionalBatchNorm2d(num_features, dim_embed),
        #     nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     #k7n3s1
        #     nn.Conv2d(num_features, out_channels, kernel_size=7, stride=1, padding=3, padding_mode=self.padding_mode)
        # )
        super().__init__()
        self.cond_batchnorm = ConditionalBatchNorm2d(num_features, dim_embed)
        self.conv0 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, device=config.DEVICE)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(num_features, out_channels, kernel_size=7, stride=1, padding=3, padding_mode=padding_mode, device=config.DEVICE)
    def forward(self, x, y_surface, y_texture):
        x = self.cond_batchnorm(x, y_surface, y_texture)
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=32, num_residuals=4, padding_mode="zeros"):
        super(Generator, self).__init__()
        self.padding_mode = padding_mode

        self.initial_down = nn.Sequential(
            #k7n32s1
            nn.Conv2d(in_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        #Down-convolution
        self.down1 = nn.Sequential(
            #k3n32s2
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n64s1
            nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.down2 = nn.Sequential(
            #k3n64s2
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=2, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n128s1
            nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        #Bottleneck: 4 residual blocks => 4 times [K3n128s1]
        self.res_blocks = [ResidualBlock(num_features*4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode) for _ in range(num_residuals)]

        #Up-convolution
        self.up1 = Up1(num_features, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)

        self.up2 = Up2(num_features, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)

        self.last = Last(num_features,out_channels, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)

    def forward(self, x, y_surface, y_texture): # B, C, 256, 256
        # y are  embed label
        x1 = self.initial_down(x) # B, 32, 256, 256
        x2 = self.down1(x1) # B, 64, 128, 128
        x = self.down2(x2) # B, 128, 64, 64
        # x = self.res_blocks(x, y_surface, y_texture) # B, 128, 64, 64
        for layer in self.res_blocks:
            x = layer(x, y_surface, y_texture)
        x = self.up1(x, y_surface, y_texture) # B, 64, 64, 64
        #Resize Bilinear
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False) # B, 64, 128, 128
        x = self.up2(x + x2, y_surface, y_texture) # B, 32, 128, 128
        #Resize Bilinear
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False) # B, 32, 256, 256
        x = self.last(x + x1, y_surface, y_texture) # B, 3, 256, 256
        #TanH
        return torch.tanh(x)

class Gray_Generator(nn.Module):
    def __init__(self, img_channels=3, out_channels=6, num_features=32, num_residuals=4, padding_mode="zeros"):
        super().__init__()
        self.padding_mode = padding_mode

        self.initial_down = nn.Sequential(
            #k7n32s1
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        #Down-convolution
        self.down1 = nn.Sequential(
            #k3n32s2
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n64s1
            nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.down2 = nn.Sequential(
            #k3n64s2
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=2, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n128s1
            nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        #Bottleneck: 4 residual blocks => 4 times [K3n128s1]
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode) for _ in range(num_residuals)]
        )

        #Up-convolution
        self.up1 = nn.Sequential(
            #k3n128s1 (should be k3n64s1?)
            nn.Conv2d(num_features*4, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.up2 = nn.Sequential(
            #k3n64s1
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #k3n64s1 (should be k3n32s1?)
            nn.Conv2d(num_features*2, num_features, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.last = nn.Sequential(
            #k3n32s1
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #k7n3s1
            nn.Conv2d(num_features, out_channels, kernel_size=7, stride=1, padding=3, padding_mode=self.padding_mode)
        )

    def forward(self, x):
        x1 = self.initial_down(x)
        x2 = self.down1(x1)
        x = self.down2(x2)
        x = self.res_blocks(x)
        x = self.up1(x)
        #Resize Bilinear
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False)
        x = self.up2(x + x2) 
        #Resize Bilinear
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False)
        x = self.last(x + x1)
        #TanH
        return torch.tanh(x)

        
def test():
    img_channels = 1
    out_channels = 2
    img_size = 4
    x = torch.randn((1, img_channels, img_size, img_size))
    gen = Gray_Generator(img_channels=img_channels, out_channels=out_channels)
    res = gen(x)
    print(res.shape)
    print(res)

if __name__ == "__main__":
    test()


