import torch
from torch import nn
import torchvision.models as models
from PIL import Image
import numpy as np
from torchvision import transforms

transform = transforms.Compose([            
 transforms.Resize(224),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 )])

def encoder_test(imgPath):
    img = Image.open(imgPath)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    vgg16 = models.vgg16(pretrained=True) 
    features = nn.Sequential(*(list(vgg16.children())[0:1]))
    #print(features)

    backbone = nn.Sequential(*features)
    backbone.eval()
    out = backbone(batch_t)
    print(out.shape)
    return out


def build_encoder():
    vgg16 = models.vgg16(pretrained=True) 
    features = nn.Sequential(*(list(vgg16.children())[0:1]))
    #print(features)

    backbone = nn.Sequential(*features)
    backbone.eval()
    
    return backbone