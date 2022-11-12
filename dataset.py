from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import config
import random

class MyDataset(Dataset):
    def __init__(self, root_A, root_B):
        self.root_A = root_A
        self.root_B = root_B

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.length_dataset = max(len(self.A_images), len(self.B_images))
        self.B_len = len(self.B_images)
        self.A_len = len(self.A_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        B_img = self.B_images[index % self.B_len]

        A_path = os.path.join(self.root_A, A_img)
        B_path = os.path.join(self.root_B, B_img)

        A_img = np.array(Image.open(A_path).convert("RGB"))
        B_img = np.array(Image.open(B_path).convert("RGB"))

        A_img = config.transform_train(image=A_img)["image"]
        B_img = config.transform_train(image=B_img)["image"]

        return A_img, B_img

class CcGANDataset(Dataset):
    # A: photos, B: Shinkai, C: Hayao
    def __init__(self, root_A, root_B, root_C):
        self.root_A = root_A
        self.root_B = root_B
        self.root_C = root_C

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.C_images = os.listdir(root_C)
        self.length_dataset = max(len(self.A_images), len(self.B_images) + len(self.C_images))
        self.B_len = len(self.B_images)
        self.A_len = len(self.A_images)
        self.C_len = len(self.A_images)
        

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        photo_img = self.A_images[index % self.A_len]
        photo_path = os.path.join(self.root_A, photo_img)
        photo_img = np.array(Image.open(photo_path).convert("RGB"))
        if random.uniform(0,1) < 0.5:
            cartoon_img = self.B_images[index % self.B_len]
            label = 0.
            cartoon_path = os.path.join(self.root_B, cartoon_img)
        else:
            cartoon_img = self.C_images[index % self.C_len]
            label = 1.
            cartoon_path = os.path.join(self.root_C, cartoon_img)
        cartoon_img = np.array(Image.open(cartoon_path).convert("RGB"))

        photo_img = config.transform_train(image=photo_img)["image"]
        cartoon_img = config.transform_train(image=cartoon_img)["image"]

        return photo_img, cartoon_img, label


class WenwenDataset(Dataset):
    def __init__(self, root_A, root_B):
        self.root_A = root_A
        self.root_B = root_B

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.length_dataset = max(len(self.A_images), len(self.B_images))
        self.B_len = len(self.B_images)
        self.A_len = len(self.A_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        B_img = self.B_images[index % self.B_len]

        A_path = os.path.join(self.root_A, A_img)
        B_path = os.path.join(self.root_B, B_img)
        Z_path = "/home/zhangyushan/kyoma/only_target/2_256.jpg"

        A_img = np.array(Image.open(A_path).convert("L"))
        B_img = np.array(Image.open(B_path).convert("L"))
        Z_img = np.array(Image.open(Z_path).convert("L"))

        A_img = config.transform_train(image=A_img)["image"]
        B_img = config.transform_train(image=B_img)["image"]
        Z_img = config.transform_train(image=Z_img)["image"]

        return A_img, B_img, Z_img

class MyTestDataset(Dataset):
    def __init__(self, root_A):
        self.root_A = root_A
        self.A_images = os.listdir(root_A)
        self.A_len = len(self.A_images)

    def __len__(self):
        return self.A_len

    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]
        A_path = os.path.join(self.root_A, A_img)
        A_img = np.array(Image.open(A_path).convert("RGB"))
        A_img = config.transform_test(image=A_img)["image"]
        return A_img, A_path