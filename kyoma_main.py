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

# Embedding
base_lr_x2y = 0.01
base_lr_y2h = 0.01

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

def get_net_y2h(extracotr_name, extractor, style_images, style_labels):
    print("Pretrain net_embed: x2h+h2y")
    print("\n Start training CNN for label embedding >>>")
    net_embed = ResNet34_embed(dim_embed=config.dim_embed)
    net_embed = net_embed.to(config.DEVICE)
    # net_embed = nn.DataParallel(net_embed)

    net_y2h = model_y2h(dim_embed=config.dim_embed)
    net_y2h = net_y2h.to(config.DEVICE)
    # net_y2h = nn.DataParallel(net_y2h)

    if not os.path.isfile(extracotr_name+"_net_embed_ckpt.pth"):
        net_embed = train_net_embed(net=net_embed, extracotr_name=extracotr_name, extractor=extractor, train_images=style_images, train_lables=style_labels, path_to_ckpt = "./saved_embed_models/")
        # save model
        torch.save({
            'net_state_dict': net_embed.state_dict(),
        }, extracotr_name+"_net_embed_ckpt.pth")
    else:
        checkpoint = torch.load(extracotr_name+"_net_embed_ckpt.pth")
        net_embed.load_state_dict(checkpoint['net_state_dict'])
        print("load net_embed from", extracotr_name+"_net_embed_ckpt.pth")

    if not os.path.isfile(extracotr_name+"_net_y2h_ckpt.pth"):
        print("\n Start training net_y2h >>>")
        unique_labels_norm = np.sort(np.array(list(set(style_labels))))
        net_y2h = train_net_y2h(unique_labels_norm, net_y2h, net_embed)
        # save model
        torch.save({
            'net_state_dict': net_y2h.state_dict(),
        }, extracotr_name+"_net_y2h_ckpt.pth")
    else:
        checkpoint = torch.load(extracotr_name+"_net_y2h_ckpt.pth")
        net_y2h.load_state_dict(checkpoint['net_state_dict'])
    return net_y2h


if __name__ == "__main__":
    style_images, style_labels, photo_images = kyoma_loder()

    extract_texture = ColorShift(config.DEVICE, mode='uniform', image_format='rgb')
    extract_surface = GuidedFilter()

    net_y2h_surface = get_net_y2h("surface", extract_surface, style_images, style_labels)
    net_y2h_texture = get_net_y2h("texture", extract_texture, style_images, style_labels)

    gen = Generator(in_channels=3)
    disc_surface = CcGAN_SAGAN_Discriminator(dim_embed=config.dim_embed)
    disc_texture = CcGAN_SAGAN_Discriminator(dim_embed=config.dim_embed)
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

    gen, disc_surface, disc_texure = train_CcGAN(args.kernel_sigma, args.kappa, photo_images, style_images, style_labels, gen, disc_surface, disc_texture, net_y2h_surface, net_y2h_texture, VGG19, save_images_folder="./saved_images/"+config.PROJECT_NAME, save_models_folder = "./saved_models/"+config.PROJECT_NAME)

    torch.save({
        'gen_state_dict': gen.state_dict(),
        'disc_surface_state_dict': disc_surface.state_dict(),
        'disc_texure_state_dict': disc_texture.state_dict(),
    }, "/home/zhangyushan/kyoma/cartoon_CcGAN/saved_models/context12/kyoma_Test_CcGAN.pth")
