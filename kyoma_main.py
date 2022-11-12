import argparse
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
import random
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import timeit
from PIL import Image
from generator_model import Generator
from discriminator_model import Discriminator
from VGGPytorch import VGGNet
import config
import cv2

from opts import parse_opts

from utils import *
from models import *
from Train_CcGAN import *

def load_images_from_dir(path): # return as numpy
    index = 0
    res = None
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            im = cv2.imread(os.path.join(root, name))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = np.expand_dims(im, axis=0)
            if index == 0:
                res = im
            else:
                res = np.concatenate((res, im), axis=0)
            index = index + 1
    return res

def kyoma_loder():
    # style0_images = load_images_from_dir("/mnt/data/ZhangYushan/scenery_cartoon/hayao/")
    # style0_labels = np.full((style0_images.shape[0]), 0.)

    # style1_images = load_images_from_dir("/mnt/data/ZhangYushan/scenery_cartoon/hosoda/")
    # style1_labels = np.full((style0_images.shape[0]), 0.5)

    # style2_images = load_images_from_dir("/mnt/data/ZhangYushan/scenery_cartoon/shinkai/")
    # style2_labels = np.full((style1_images.shape[0]), 1.)

    # style_images = np.concatenate((style0_images, style1_images, style2_images), axis=0)
    # style_labels = np.concatenate((style0_labels, style1_labels, style2_labels), axis=0)

    # with open("/home/zhangyushan/kyoma/cartoon_CcGAN/data/hayao_hosoda_shinkai.npy", "wb") as f:
    #     np.save(f, style_images)
    # with open("/home/zhangyushan/kyoma/cartoon_CcGAN/data/hayao_hosoda_shinkai_labels.npy", "wb") as f:
    #     np.save(f, style_labels)

    # photo_images = load_images_from_dir("/home/zhangyushan/kyoma/cartoon_CcGAN/data/photo")

    with open("/home/zhangyushan/kyoma/cartoon_CcGAN/data/hayao_hosoda_shinkai.npy", "rb") as f:
        style_images = np.load(f)
    print("loaded style images")
    
    with open("/home/zhangyushan/kyoma/cartoon_CcGAN/data/hayao_hosoda_shinkai_labels.npy", "rb") as f:
        style_labels = np.load(f)
    print("loaded style labels")

    with open("/home/zhangyushan/kyoma/cartoon_CcGAN/data/landscape_photos.npy", "rb") as f:
        photo_images = np.load(f)
    print("loaded photos")

    return style_images, style_labels, photo_images

if __name__ == "__main__":
    gen = Generator(in_channels=5)
    disc_surface = cont_cond_cnn_discriminator()
    disc_texture = cont_cond_cnn_discriminator()
    gen = nn.DataParallel(gen)
    disc_surface = nn.DataParallel(disc_surface)
    disc_texture = nn.DataParallel(disc_texture)
    
    VGG19 = VGGNet(in_channels=3, VGGtype="VGG19", init_weights=config.VGG_WEIGHTS, batch_norm=False, feature_mode=True)
    VGG19 = VGG19.to(config.DEVICE)
    VGG19.eval()

    style_images, style_labels, photo_images = kyoma_loder()

    if args.kernel_sigma<0:
        std_label = np.std(style_labels)
        args.kernel_sigma =1.06*std_label*(len(style_labels))**(-1/5)
        print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
        print("\n The std of {} labels is {} so the kernel sigma is {}".format(len(style_labels), std_label, args.kernel_sigma))
    
    unique_labels_norm = np.sort(np.array(list(set(style_labels))))
    if args.kappa<0:
        n_unique = len(unique_labels_norm)

        diff_list = []
        for i in range(1,n_unique):
            diff_list.append(unique_labels_norm[i] - unique_labels_norm[i-1])
        kappa_base = np.abs(args.kappa)*np.max(np.array(diff_list))

        if args.threshold_type=="hard":
            args.kappa = kappa_base
        else:
            args.kappa = 1/kappa_base**2

    gen, disc_surface, disc_texure = train_CcGAN(args.kernel_sigma, args.kappa, photo_images, style_images, style_labels, gen, disc_surface, disc_texture, VGG19, save_images_folder="/home/zhangyushan/kyoma/cartoon_CcGAN/saved_images/surface_texture/", save_models_folder = "/home/zhangyushan/kyoma/cartoon_CcGAN/saved_models/surface_texture/")

    torch.save({
        'gen_state_dict': gen.state_dict(),
        'disc_surface_state_dict': disc_surface.state_dict(),
        'disc_texure_state_dict': disc_texture.state_dict(),
    }, "/home/zhangyushan/kyoma/cartoon_CcGAN/saved_models/surface_texture/kyoma_Test_CcGAN.pth")
