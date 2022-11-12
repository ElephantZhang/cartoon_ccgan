import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch implementation by vinesmsuic
# Referenced from official tensorflow implementation: https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/train_code/network.py
# slim.convolution2d uses constant padding (zeros).


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, padding_mode):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
        )

    def forward(self, x):
        #Elementwise Sum (ES)
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=32, num_residuals=4, padding_mode="zeros"):
        super().__init__()
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

    def forward(self, z, y_surface, y_texture):
        # y is condition
        y_surface = y_surface.repeat_interleave(256*256).reshape(-1,1,256,256)
        y_texture = y_texture.repeat_interleave(256*256).reshape(-1,1,256,256)
        x = torch.cat((z,y_surface, y_texture), 1)
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


