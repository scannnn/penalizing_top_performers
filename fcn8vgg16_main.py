import torch
from util.logger import Logger
import utils
from configs import cfg
from torch.utils.data import DataLoader
from train_fcn8 import *

source_img_path = "/content/gdrive/MyDrive/AI_PROJECT(BLG_527E)/dataset/source/images"
source_label_path ="/content/gdrive/MyDrive/AI_PROJECT(BLG_527E)/dataset/source/labels"
target_img_path = "/content/gdrive/MyDrive/AI_PROJECT(BLG_527E)/dataset/target/images"


""" 
MAIN TRAIN CODE STARTS HERE FOR FCN8 WITH VGG16 BACKBONE
"""
if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    n_classes = 19
    num_epochs = 100
    pretrained = True
    fixed_feature = False
    epoch = 4

    logger = Logger(model_name="fcn8_vgg16", data_name='gta5')

    src_train_imgs = utils.read_path(source_img_path,"png")
    src_train_labels = utils.read_path(source_label_path,"png")
    target_train_imgs = utils.read_path(target_img_path,"jpg")

    src_train_ds = utils.GTA5Dataset(cfg, src_train_imgs, src_train_labels)
    trgt_train_ds = utils.Dataset(cfg, target_train_imgs)



    src_train_loader = DataLoader(src_train_ds, batch_size=1, shuffle=True, drop_last=True)
    tgt_train_loader = DataLoader(trgt_train_ds, batch_size=1, shuffle=True, drop_last=True)

    ### Model
    model = fcn8_vgg16(n_classes)
    model.to(device)

    ### Load model
    # logger.load_model(model, 'epoch_3')

    ### Optimizers
    if pretrained and fixed_feature: #fine tunning
        params_to_update = model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adadelta(params_to_update)
    else:
        optimizer = torch.optim.Adadelta(model.parameters())

    ### Train
    # If model is loaded then epoch must be the number that we saved lastly
    trainer = Trainer(model, optimizer, logger, num_epochs, src_train_loader, epoch=0)
    trainer.train()
