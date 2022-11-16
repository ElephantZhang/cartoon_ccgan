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
from models.ResNet_embed import ResNet34_embed
from models.ResNet_embed import model_y2h
from models.CcGAN_SAGAN import CcGAN_SAGAN_Discriminator
from utils import *
from models import *
from Train_CcGAN import *
from train_net_for_label_embed import train_net_embed
from train_net_for_label_embed import train_net_y2h
from surface_extractor import GuidedFilter
from texture_extractor import ColorShift

args = parse_opts()
config.DEVICE = args.cuda
config.PROJECT_NAME = args.project_name
config.BATCH_SIZE = args.batch_size_disc
config.LAMBDA_SURFACE = args.lambda_surface
config.LAMBDA_TEXTURE = args.lambda_texture
config.LAMBDA_CONTENT = args.lambda_context

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

def date_creator():
    style0_images = load_images_from_dir("/home/zhangyushan/kyoma/cartoon_ccgan/data/hayao")
    style0_labels = np.full((style0_images.shape[0]), 0.)
    print("loaded hayao")

    with open("./data/isometric_hayao.npy", "wb") as f:
        np.save(f, style0_images)
    with open("./data/isometric_hayao_labels.npy", "wb") as f:
        np.save(f, style0_labels)

    style1_images = load_images_from_dir("/home/zhangyushan/kyoma/cartoon_ccgan/data/hosoda")
    style1_labels = np.full((style1_images.shape[0]), 0.5)
    print("loaded hosoda")
    
    with open("./data/isometric_hosoda.npy", "wb") as f:
        np.save(f, style1_images)
    with open("./data/isometric_hosoda_labels.npy", "wb") as f:
        np.save(f, style1_labels)

    style2_images = load_images_from_dir("/home/zhangyushan/kyoma/cartoon_ccgan/data/shinkai")
    style2_labels = np.full((style2_images.shape[0]), 1.)
    print("loaded shinkai")

    with open("./data/isometric_shinkai.npy", "wb") as f:
        np.save(f, style2_images)
    with open("./data/isometric_shinkai_labels.npy", "wb") as f:
        np.save(f, style2_labels)

    # style_images = np.concatenate((style0_images, style1_images, style2_images), axis=0)
    # style_labels = np.concatenate((style0_labels, style1_labels, style2_labels), axis=0)

    # with open("./data/isometric_hayao_hosoda_shinkai.npy", "wb") as f:
    #     np.save(f, style_images)
    # with open("./data/isometric_hayao_hosoda_shinkai_labels.npy", "wb") as f:
    #     np.save(f, style_labels)

def kyoma_loder():
    with open("./data/isometric_hayao.npy", "rb") as f:
        style0_images = np.load(f)
    with open("./data/isometric_shinkai.npy", "rb") as f:
        style2_images = np.load(f)
    style_images = np.concatenate((style0_images, style2_images), axis=0)

    with open("./data/isometric_hayao_labels.npy", "rb") as f:
        style0_labels = np.load(f)
    with open("./data/isometric_shinkai_labels.npy", "rb") as f:
        style2_labels = np.load(f)
    style_labels = np.concatenate((style0_labels, style2_labels), axis=0)

    with open("./data/landscape_photos.npy", "rb") as f:
        photo_images = np.load(f)
    print("loaded photos")

    return style_images, style_labels, photo_images

if __name__ == "__main__":
    style_images, style_labels, photo_images = kyoma_loder()

    gen = Generator(img_channels=4)
    disc_surface = cont_cond_cnn_discriminator()
    # gen = nn.DataParallel(gen)
    # disc_surface = nn.DataParallel(disc_surface)
    # disc_texture = nn.DataParallel(disc_texture)
    
    VGG19 = VGGNet(in_channels=3, VGGtype="VGG19", init_weights=config.VGG_WEIGHTS, batch_norm=False, feature_mode=True)
    VGG19 = VGG19.to(config.DEVICE)
    VGG19.eval()

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

    gen, disc_surface, disc_texure = train_CcGAN(args.kernel_sigma, args.kappa, photo_images, style_images, style_labels, gen, disc_surface, VGG19, save_images_folder="./saved_images/"+config.PROJECT_NAME, save_models_folder = "./saved_models/"+config.PROJECT_NAME)

    torch.save({
        'gen_state_dict': gen.state_dict(),
        'disc_surface_state_dict': disc_surface.state_dict(),
    }, "./saved_models/" + config.PROJECT_NAME + "/kyoma_Test_CcGAN.pth")
