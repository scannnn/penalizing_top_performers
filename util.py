import numpy as np 
import matplotlib.pyplot as plt 
import os, time, pickle, json 
from glob import glob 
from PIL import Image
import cv2 
from typing import List, Tuple, Dict
from statistics import mean 
from tqdm import tqdm 

import torch 
import torch.nn as nn 
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader 
import torch.nn.functional as F

MEAN = (0.5, 0.5, 0.5,)
STD = (0.5, 0.5, 0.5,)
RESIZE = 512

def read_path(source_path, ext) -> List[str]:
    root_path = "."
    path = os.path.join(root_path, source_path)
    dataset = []
    for p in glob(path+"/"+"*."+ext):
        dataset.append(p)
    return dataset 

class Transform():
    def __init__(self, resize=RESIZE, mean=MEAN, std=STD):
        self.data_transform = transforms.Compose([
            transforms.Resize((resize, resize)), 
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ])
        
    def __call__(self, img: Image.Image):
        return self.data_transform(img)


class Dataset(object):
    
    def __init__(self, files: List[str]):
        self.files = files 
        self.trasformer = Transform()
        
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx])
        input_tensor = self.trasformer(img)
        return input_tensor
    # def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    #     img = Image.open(self.files[idx])
    #     input, output = self._separate(img)
    #     input.save("./dataset/target/one/"+str(self.a)+".jpg")
    #     self.a = self.a+1
    #     input_tensor = self.trasformer(input)
    #     output_tensor = self.trasformer(output)
    #     return input_tensor, output_tensor 
    
    def __len__(self):
        return len(self.files)

# class Target_Dataset(Dataset):
#     def _separate(self, img) -> Tuple[Image.Image, Image.Image]:
#         img = np.array(img, dtype=np.uint8)
#         h, w, _ = img.shape
#         w = int(w/2)
#         return Image.fromarray(img[:, :w, :]) , Image.fromarray(img[:, w:, :])
        
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         img = Image.open(self.files[idx])
#         inp, _ = self._separate(img)
#         input_tensor = self.trasformer(inp)
#         return input_tensor
    
#     def __len__(self):
#         return len(self.files)

# def show_img_source(img: torch.Tensor, img1: torch.Tensor):
#     fig, axes = plt.subplots(1, 2, figsize=(15, 8))
#     ax = axes.ravel()
#     ax[0].imshow(img.permute(1, 2, 0))
#     ax[0].set_xticks([])
#     ax[0].set_yticks([])
#     ax[0].set_title("source input", c="g")
#     ax[1].imshow(img1.permute(1, 2, 0))
#     ax[1].set_xticks([])
#     ax[1].set_yticks([])
#     ax[1].set_title("source label", c="g")
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.show()        

def show_img(img: torch.Tensor,):
    p = plt.imshow(img.permute(1,2,0))
    plt.show() 

def adjust_learning_rate(method, base_lr, iters, max_iters, power):
    if method=='poly':
        lr = base_lr * ((1 - float(iters) / max_iters) ** (power))
    else:
        raise NotImplementedError
    return lr

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))