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
import transform

MEAN = (0.5, 0.5, 0.5,)
STD = (0.5, 0.5, 0.5,)
RESIZE = 512

def build_transform(cfg, mode, is_source):
    if mode=="train":
        w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN if is_source else cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
        trans_list = [
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ]
        if cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN > 0:
            trans_list = [transform.RandomHorizontalFlip(p=cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN),] + trans_list
        if cfg.INPUT.INPUT_SCALES_TRAIN[0]==cfg.INPUT.INPUT_SCALES_TRAIN[1] and cfg.INPUT.INPUT_SCALES_TRAIN[0]==1:
            trans_list = [transform.Resize((h, w)),] + trans_list
        else:
            trans_list = [
                transform.RandomScale(scale=cfg.INPUT.INPUT_SCALES_TRAIN),
                transform.RandomCrop(size=(h, w), pad_if_needed=True),
            ] + trans_list
        if is_source:
            trans_list = [
                transform.ColorJitter(
                    brightness=cfg.INPUT.BRIGHTNESS,
                    contrast=cfg.INPUT.CONTRAST,
                    saturation=cfg.INPUT.SATURATION,
                    hue=cfg.INPUT.HUE,
                ),
            ] + trans_list
        trans = transform.Compose(trans_list)
    else:
        w, h = cfg.INPUT.INPUT_SIZE_TEST
        trans = transform.Compose([
            transform.Resize((h, w), resize_label=False),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
    return trans

def read_path(source_path, ext) -> List[str]:
    root_path = "/content "
    path = os.path.join(root_path, source_path)
    dataset = []
    for p in glob(path+"/"+"*."+ext):
        dataset.append(p)
    dataset.sort()
    return dataset 

class Transform():
    def __init__(self, resize=RESIZE, mean=MEAN, std=STD):
        self.data_transform = transforms.Compose([
            transforms.Resize((resize, resize)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
        
    def __call__(self, img: Image.Image):
        return self.data_transform(img)

class GTA5Dataset(object):
    
    def __init__(self, cfg, img_files: List[str], label_files : List[str]):
        self.img_files = img_files[:50]
        self.label_files = label_files[:50]
        self.trasformer = build_transform(cfg, "train", True)

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        #print(self.img_files[idx], "->", self.label_files[idx])
        img = Image.open(self.img_files[idx]).convert('RGB')
        label = np.array(Image.open(self.label_files[idx]),dtype=np.float32)

        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)
        img_tens, label_tens  = self.trasformer(img, label)
        
        return img_tens, label_tens
    
    def __len__(self):
        return len(self.img_files)

class Dataset(object):
    
    def __init__(self, cfg, files: List[str]):
        self.files = files 
        self.trasformer = build_transform(cfg, "train", False)#Transform()
        
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx]).convert('RGB')
        input_tensor, _ = self.trasformer(img, img)
        return input_tensor
    
    def __len__(self):
        return len(self.files)
      

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