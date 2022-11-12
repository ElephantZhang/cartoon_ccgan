from re import L
from matplotlib.font_manager import weight_dict
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import math
from PIL import Image
from torchvision import transforms
import pytorch_colors as colors

from skimage.util import img_as_ubyte
import skimage.transform as tf
from surface_extractor import GuidedFilter

import model_hed_sketch
import cv2
from torchvision.utils import save_image
# from line_disstiller_model import Net as LineDistillerNet

class ColorShift():
    def __init__(self, device: torch.device='cpu', mode='uniform', image_format='rgb'):
        self.dist: torch.distributions = None
        self.dist_param1: torch.Tensor = None
        self.dist_param2: torch.Tensor = None

        if(mode == 'uniform'):
            self.dist_param1 = torch.tensor((0.199, 0.487, 0.014), device=device)
            self.dist_param2 = torch.tensor((0.399, 0.687, 0.214), device=device)
            if(image_format == 'bgr'):
                self.dist_param1 = torch.permute(self.dist_param1, (2, 1, 0))
                self.dist_param2 = torch.permute(self.dist_param2, (2, 1, 0))

            self.dist = torch.distributions.Uniform(low=self.dist_param1, high=self.dist_param2)
            
        elif(mode == 'normal'):
            self.dist_param1 = torch.tensor((0.299, 0.587, 0.114), device=device)
            self.dist_param2 = torch.tensor((0.1, 0.1, 0.1), device=device)
            if(image_format == 'bgr'):
                self.dist_param1 = torch.permute(self.dist_param1, (2, 1, 0))
                self.dist_param2 = torch.permute(self.dist_param2, (2, 1, 0))

            self.dist = torch.distributions.Normal(loc=self.dist_param1, scale=self.dist_param2)
        
    #Allow taking mutiple images batches as input
    #So we can do: gray_fake, gray_cartoon = ColorShift(output, input_cartoon)
    def process(self, *image_batches: torch.Tensor):  # The shape of image_batches is (B, 3, H, W)  (B, 3, H, W) -> (B, 1, H, W) ?
        # Sample the random color shift coefficients
        weights = self.dist.sample()

        # images * weights[None, :, None, None] => Apply weights to r,g,b channels of each images
        # torch.sum(, dim=1) => Sum along the channels so (B, 3, H, W) become (B, H, W)
        # .unsqueeze(1) => add back the channel so (B, H, W) become (B, 1, H, W)
        # .repeat(1, 3, 1, 1) => (B, 1, H, W) become (B, 3, H, W) again
        return ((((torch.sum(images * weights[None, :, None, None], dim= 1)) / weights.sum()).unsqueeze(1)).repeat(1, 3, 1, 1) for images in image_batches)

class DoG():
    def __init__(self, device: torch.device='cpu', mode='uniform', image_format='rgb'):
        self.device = device

    def get_gaussian_kernel(self, kernel_size=5, sigma=2, channels=2):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        return gaussian_kernel

    def process(self, *image_batches: torch.Tensor):
        sigma = 1
        phie = 5
        tau = 0.981
        kernel_size = 15

        kernel_e = self.get_gaussian_kernel(kernel_size=kernel_size, sigma=sigma, channels=1).to(self.device)
        kernel_r = self.get_gaussian_kernel(kernel_size=kernel_size, sigma=sigma*1.5, channels=1).to(self.device)

        dog_images = []
        for images in image_batches:
            with torch.no_grad():
                images_lab = colors.rgb_to_lab(images)
                L = images_lab[:, 0, :, :]
                L = L.unsqueeze(1)
                imge = torch.conv2d(L, kernel_e, padding=int(kernel_size/2))
                imgr = torch.conv2d(L, kernel_r, padding=int(kernel_size/2))
                res = torch.tanh((imge - tau * imgr) * phie) + 1
                res[res > 1] = 1.0
                dog_images.append(res.repeat(1, 3, 1, 1))
        return tuple(dog_images)
        # return tuple(dog_images)[0]


# class Sobel(nn.Module):
#     def __init__(self, device: torch.device='cpu', mode='uniform', image_format='rgb'):
#         self.filter = torch.tensor([
#             [
#                 [2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]
#             ],
#             [
#                 [2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]
#             ]

#         ]).float().to(device)
#         self.filter = torch.unsqueeze(self.filter, dim=1)


#     def process(self, *image_batches: torch.Tensor):
#         t = transforms.Grayscale(1)
#         sobel_images = []
#         for images in image_batches:
#             with torch.no_grad():
#                 images = t(images)
#                 res = torch.conv2d(images, self.filter, padding=1)
#                 res = torch.mul(res, res)
#                 res = torch.sum(res, dim=1, keepdim=True)
#                 res = torch.sqrt(res)
#                 res = res / res.max()
#                 sobel_images.append(res.repeat(1, 3, 1, 1))
#         return tuple(sobel_images)

# class LineDistiller():
#     def __init__(self, device: torch.device='cpu', mode='uniform', image_format='rgb'):
#         model_path = "./line_distiller_model.pth"
#         self.model = LineDistillerNet().to(device)
#         self.model.load_state_dict(torch.load(model_path, map_location=device))
#         self.model.eval()
#         self.preprocess = transforms.Compose([
#             transforms.Normalize(
#                 mean=[0.5, 0.5, 0.5],
#                 std=[0.5, 0.5, 0.5]),
#         ])
    
#     def process(self, *image_batches: torch.Tensor):
#         res_list = []
#         with torch.no_grad():
#             for images in image_batches:
#                 images = self.preprocess(images)
#                 res = self.model(images)
#                 res_list.append(res.repeat(1, 3, 1, 1))
#         return tuple(res_list)


# class Sketch():
#     def __init__(self, device: torch.device='cpu', mode='uniform', image_format='rgb'):
#         self.device = device
#         self.sktNetwork = model_hed_sketch.Network().to(self.device).eval()


#     def process(self, *image_batches: torch.Tensor):
#         sketch_images = []

#         for images in image_batches:
#             with torch.no_grad():
#                 inImg = (images * 0.5 + 0.5) * 255
#                 ouImg = self.sktNetwork(inImg)
#                 ouImg = 1.0 - ouImg.clamp(0.0, 1.0)
#                 ouImg[ouImg<1] = 0
#                 sketch_images.append(ouImg.repeat(1, 3, 1, 1))
#         return tuple(sketch_images)

# class DoNothing():
#     def __init__(self, device: torch.device='cpu', mode='uniform', image_format='rgb'):
#         self.device = device

#     def process(self, x: torch.Tensor, y: torch.Tensor, r, eps=1e-2):
#         return x


# def read_img(img_path):
#     t = transforms.Compose([
#         # transforms.Grayscale(1),
#         transforms.ToTensor(),
#     ])
#     pil_image = Image.open(img_path)
#     if pil_image.mode == "RGBA":
#         image = Image.new("RGB", pil_image.size, (255,255,255))
#         image.paste(pil_image, mask=pil_image.split()[3])
#     else:
#         image = pil_image.convert('RGB')
#     return t(image)

# def main():
#     # images = torch.randn(5, 3, 256, 256)  # dtype = float
#     # model = Sobel()

#     # with torch.no_grad():
#     #     output = model(images)
#     #     print(output)

#     lenna = read_img("/home/zhangyushan/kyoma/White-box-Cartoonization-PyTorch/data/train/cartoon/44.png")
#     batch_images = torch.unsqueeze(lenna, 0)
#     print(batch_images.shape)
#     model = DoG()
#     with torch.no_grad():
#         output = model.process(batch_images)[0]
#         # output = output.numpy()
#         # formatted = (255 - output * 255 / np.max(output)).astype('uint8')
#         # formatted *= 255
#         # formatted = 255 - formatted
#         # output = output[0, :, :, :]
#         # cv2.imwrite("/home/zhangyushan/kyoma/lenna_dog_before.png", output * 255)
#         a = torch.cat([batch_images, output, output], axis=3)
#         save_image(a, "/home/zhangyushan/kyoma/lenna_dog_before.png")
#         # Image.fromarray(output, mode='RGB').save("/home/zhangyushan/kyoma/lenna_dog_before.png")
#         # output.save("/home/zhangyushan/lenna_sobel.png", quality=95)

# if __name__ == '__main__':
#     main()



# # if __name__ == "__main__":
# #     color_shift = ColorShift()
# #     input1 = torch.randn(5,3,256,256)
# #     input2 = torch.randn(5,3,256,256)
# #     result1, result2 = color_shift.process(input1, input2)
# #     print(result1.shape, result2.shape) #torch.Size([5, 3, 256, 256]) torch.Size([5, 3, 256, 256])