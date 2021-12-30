from discriminator import build_discriminator, build_pixel_discriminator
from encoder import build_encoder
from classifier import build_classifier
from generator import build_generator
from util import adjust_learning_rate, soft_label_cross_entropy
import util
import torch.nn.functional as F
import numpy as np 
import os
from statistics import mean 
from tqdm import tqdm 
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms

def train():
    feature_extractor = build_encoder()
    device = torch.device("cpu")
    feature_extractor.to(device)
    
    classifier = build_classifier()
    classifier.to(device)

    generator = build_generator()
    generator.to(device)

    discriminator = build_discriminator()
    
    model_D = build_pixel_discriminator()
    model_D.to(device)

    batch_size = 1

    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
    optimizer_fea.zero_grad()
    
    optimizer_gn = torch.optim.SGD(classifier.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0005)
    optimizer_gn.zero_grad()
    
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.0008, betas=(0.9, 0.999))
    optimizer_D.zero_grad()

    output_dir = "models"

    start_epoch = 0
    iteration = 0
    MAX_ITERATION = 100
    src_train_imgs = util.read_path("dataset/source/images","png")
    src_train_labels = util.read_path( "dataset/source/labels","png")
    target_train_imgs = util.read_path("dataset/target","jpg")

    src_train_imgs_ds = util.Dataset(src_train_imgs)
    src_train_labels_ds = util.Dataset(src_train_labels)
    trgt_train_ds = util.Dataset(target_train_imgs)

    BATCH_SIZE = 1
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    src_train_img_loader = DataLoader(src_train_imgs_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    src_train_label_loader = DataLoader(src_train_labels_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    tgt_train_loader = DataLoader(trgt_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = torch.nn.BCELoss(reduction='none')

    feature_extractor.train()
    generator.train()
    model_D.train()

    
    for i, (src_input, src_label, tgt_input) in enumerate(zip(tqdm(src_train_img_loader), tqdm(src_train_label_loader), tqdm(tgt_train_loader))):
        current_lr = adjust_learning_rate("poly", 0.002, iteration, MAX_ITERATION, power=0.9)
        current_lr_D = adjust_learning_rate("poly", 0.02, iteration, MAX_ITERATION, power=0.9)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_gn.param_groups)):
            optimizer_gn.param_groups[index]['lr'] = current_lr*10
        for index in range(len(optimizer_D.param_groups)):
            optimizer_D.param_groups[index]['lr'] = current_lr_D
#######################
        src_input = src_input[0]
        src_label = src_label[0]
        tgt_input = tgt_input[0]
        transform = transforms.Compose([            
            transforms.Resize(224),                    
            transforms.CenterCrop(224),                
            transforms.ToTensor(),                     
            transforms.Normalize(                      
            mean=[0.485, 0.456, 0.406],                
            std=[0.229, 0.224, 0.225]                  
        )])
        img_src_input = transform(src_input.permute(1,2,0))
        img_src_label = transform(src_label.permute(1,2,0))
        img_tgt_input = tgt_input(tgt_input.permute(1,2,0))
        src_input_r = torch.unsqueeze(img_src_input, 0)
        src_label_r = torch.unsqueeze(img_src_label, 0)
        tgt_input_r = torch.unsqueeze(img_tgt_input, 0)

###################
        optimizer_fea.zero_grad()
        optimizer_gn.zero_grad()
        optimizer_D.zero_grad()
        
        # src_input = src_input.cuda(non_blocking=True)
        # src_label = src_label.cuda(non_blocking=True).long()
        # tgt_input_r = tgt_input_r.cuda(non_blocking=True)

        src_size = src_input_r.shape[-2:]
        tgt_size = tgt_input_r.shape[-2:]

        src_fea = feature_extractor(src_input_r)
        src_pred = classifier(src_fea, src_size)
        # src_pred ile bir loss hesaplayacağız ve bu loss ile backward yapacağız
        temperature = 1.8
        src_pred = src_pred.div(temperature)
    
        print(src_pred.shape)
        

        loss_seg = criterion(src_pred, src_label_r)
        loss_seg.backward()
        

        src_soft_label = F.softmax(src_pred, dim=1).detach()
        src_soft_label[src_soft_label>0.9] = 0.9
        
        tgt_fea = feature_extractor(tgt_input_r)
        tgt_pred = generator(tgt_fea)
        tgt_soft_label = F.softmax(tgt_pred, dim=1)
        
        tgt_soft_label = tgt_soft_label.detach()
        tgt_soft_label[tgt_soft_label>0.9] = 0.9
        
        tgt_D_pred = model_D(tgt_fea, tgt_size)
        loss_adv_tgt = 0.001*soft_label_cross_entropy(tgt_D_pred, torch.cat((tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))
        loss_adv_tgt.backward()

        optimizer_fea.step()
        optimizer_gn.step()

        optimizer_D.zero_grad()
        
        src_D_pred = model_D(src_fea.detach(), src_size)
        loss_D_src = 0.5*soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1))
        loss_D_src.backward()

        tgt_D_pred = model_D(tgt_fea.detach(), tgt_size)
        loss_D_tgt = 0.5*soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tgt_soft_label), tgt_soft_label), dim=1))
        loss_D_tgt.backward()

        optimizer_D.step()
            
        
        iteration = iteration + 1

        n = src_input_r.size(0)

                
        if (iteration == MAX_ITERATION or iteration % 5==0):
            filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(), 'classifier':classifier.state_dict(), 'model_D': model_D.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(), 'optimizer_gn': optimizer_gn.state_dict(), 'optimizer_D': optimizer_D.state_dict()}, filename)

        if iteration == MAX_ITERATION:
            break
        