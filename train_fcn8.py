from __future__ import absolute_import, division, print_function
from discriminator import discriminator
import utils
from configs import cfg
from torch.utils.data import DataLoader
from util.logger import Logger
import torch
from segmentation.trainer import Trainer
from segmentation.predict import *
from torchvision.models import vgg16
from torch import nn
from dataclasses import dataclass


@dataclass
class CopyFeatureInfo:
	index: int
	out_channels: int

class SqueezeExtractor(nn.Module):
	def __init__(self, model, features, fixed_feature=True):
		super(SqueezeExtractor, self).__init__()
		self.model = model
		self.features = features
		if fixed_feature:
			for param in self.features.parameters():
				param.requires_grad = False

	def get_copy_feature_info(self):
		"""
		Get [CopyFeatureInfo] when sampling such as maxpooling or conv2d which has the 2x2 stride.
		:return: list. [CopyFeatureInfo]
		"""
		raise NotImplementedError()

	def _get_last_conv2d_out_channels(self, features):
		for idx, m in reversed(list(enumerate(features.modules()))):
			if isinstance(m, nn.Conv2d):
				return int(m.out_channels)
		assert False

class _VGG(SqueezeExtractor):
	def __init__(self, model, features, fixed_feature=True):
		super(_VGG, self).__init__(model, features, fixed_feature)

	def get_copy_feature_info(self):

		lst_copy_feature_info = []
		for i in range(len(self.features)):
			if isinstance(self.features[i], nn.MaxPool2d):
				out_channels = self._get_last_conv2d_out_channels(self.features[:i])
				lst_copy_feature_info.append(CopyFeatureInfo(i, out_channels))
		return lst_copy_feature_info

def vgg_16():
    model = vgg16()
    model.load_state_dict(torch.load("/Users/can.cetindag/Documents/PERSONAL/AI/PROJECT/penalizing_top_performers/models/vgg16-397923af.pth"))
    features = nn.Sequential(*(list(model.children())[0:1]))
    backbone = nn.Sequential(*features)
    return backbone

def vgg_16_v2():
    model = vgg16(pretrained=True)
    #model.load_state_dict(torch.load("/Users/can.cetindag/Documents/PERSONAL/AI/PROJECT/penalizing_top_performers/models/vgg16-397923af.pth"))
    return _VGG(model, model.features, True)

class FCN8V2(nn.Module):

    def __init__(self, n_classes, pretrained_model: SqueezeExtractor):
        super(FCN8V2, self).__init__()
        self.features = pretrained_model.features
        self.copy_feature_info = pretrained_model.get_copy_feature_info()
        self.score_pool3 = nn.Conv2d(self.copy_feature_info[-3].out_channels,
                                     n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(self.copy_feature_info[-2].out_channels,
                                     n_classes, kernel_size=1)

        self.upsampling2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4,
                                              stride=2, bias=False)
        self.upsampling8 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=16,
                                              stride=8, bias=False)

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                channels = m.out_channels

        self.classifier = nn.Sequential(nn.Conv2d(channels, n_classes, kernel_size=1), nn.Sigmoid())
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        saved_pools = []
        pool5_out = None
        o = x
        for i in range(len(self.features)):
            o = self.features[i](o)
            if i == self.copy_feature_info[-3].index or\
                    i == self.copy_feature_info[-2].index:
                saved_pools.append(o)
            if i == self.copy_feature_info[-1].index:
                pool5_out = self.features[i](o)

        o = self.classifier(o)
        o = self.upsampling2(o)

        o2 = self.score_pool4(saved_pools[1])
        o = o[:, :, 1:1 + o2.size()[2], 1:1 + o2.size()[3]]
        o = o + o2

        o = self.upsampling2(o)

        o2 = self.score_pool3(saved_pools[0])
        o = o[:, :, 1:1 + o2.size()[2], 1:1 + o2.size()[3]]
        o = o + o2

        o = self.upsampling8(o)
        cx = int((o.shape[3] - x.shape[3]) / 2)
        cy = int((o.shape[2] - x.shape[2]) / 2)
        o = o[:, :, cy:cy + x.shape[2], cx:cx + x.shape[3]]

        return [pool5_out, o]


class GENERATOR(nn.Module):

    def __init__(self):
        super(GENERATOR, self).__init__()

        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, 
                     stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, 
                     stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, 
                     stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)

        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.in1 = nn.InstanceNorm2d(256)
        self.leakyRelu = nn.LeakyReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.in2 = nn.InstanceNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.in3 = nn.InstanceNorm2d(64)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out = self.bn1(out1)
        out = self.relu(out)
        out2 = self.relu(self.bn2(self.conv2(out)))
        out = self.bn2(out2)
        out = self.relu(out)
        out = self.deconv1(out)

        d1 = self.in1(self.leakyRelu(self.deconv1(out)))
        d2 = self.in2(self.leakyRelu(self.deconv2(d1)))
        d3 = self.in3(self.leakyRelu(self.deconv3(d2)))           

        return d3


class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x

class PatchGAN(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 128, batchnorm=False)
        self.d2 = DownSampleConv(128, 256)
        self.d3 = DownSampleConv(256, 512)
        self.d4 = DownSampleConv(512, 1024)
        self.d5 = DownSampleConv(1024, 2048)
        self.final = nn.Conv2d(2048, 1, kernel_size=1)

        self._initialize_weights()

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        x4 = self.d5(x3)
        xn = self.final(x4)
        return xn

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def fcn8_vgg16_V2(n_classes):
    return FCN8V2(n_classes, vgg_16_v2())

def main(v2=True, device='cuda'):
    source_img_path = "gdrive/MyDrive/AI_PROJECT(BLG_527E)/dataset/source/images"
    source_label_path ="gdrive/MyDrive/AI_PROJECT(BLG_527E)/dataset/source/labels"
    target_img_path = "gdrive/MyDrive/AI_PROJECT(BLG_527E)/dataset/target"
    src_train_imgs = utils.read_path(source_img_path,"png")
    src_train_labels = utils.read_path(source_label_path,"png")
    target_train_imgs = utils.read_path(target_img_path,"jpg")

    src_train_ds = utils.GTA5Dataset(cfg, src_train_imgs, src_train_labels)
    trgt_train_ds = utils.Dataset(cfg, target_train_imgs)

    n_classes = 19
    num_epochs = 100
    pretrained = True
    fixed_feature = False

    logger = Logger(model_name="fcn8_vgg16", data_name='gta5')

    src_train_loader = DataLoader(src_train_ds, batch_size=1, shuffle=False, drop_last=True)
    tgt_train_loader = DataLoader(trgt_train_ds, batch_size=1, shuffle=True, drop_last=True)

    ### Model
    # encoder_model = vgg_16()
    classifier_model = fcn8_vgg16_V2(n_classes=n_classes)
    generator_model = GENERATOR()
    # TODO: DISCRIMINATOR MODEL YAZILACAK
    discriminator_model = PatchGAN()
    # encoder_model.to(device)
    classifier_model.to(device)
    generator_model.to(device)
    discriminator_model.to(device)

    optimizer_cl = torch.optim.SGD(classifier_model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
    optimizer_cl.zero_grad()

    optimizer_gn = torch.optim.SGD(generator_model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
    optimizer_gn.zero_grad()

    optimizer_d = torch.optim.Adam(discriminator_model.parameters(), lr=0.002, betas=(0.9, 0.99))
    optimizer_d.zero_grad()


    ###Load model
    ###please check the foloder: (.segmentation/test/runs/models)
    #logger.load_model(model, 'epoch_15')

    ### Optimizers
    if pretrained and fixed_feature: #fine tunning
        params_to_update = classifier_model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in classifier_model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adadelta(params_to_update)
    else:
        optimizer = torch.optim.Adadelta(classifier_model.parameters())

    ### Train
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainer = Trainer(classifier_model, generator_model, discriminator_model, optimizer_cl, optimizer_gn, optimizer_d, logger, num_epochs, src_train_loader, tgt_train_loader)
    trainer.train()


    #### Wariting the predict result.
    """predict(model, 
            'dataset/cityspaces/input.png', 
            'dataset/cityspaces/output.png')""" 



generator = GENERATOR()
print(generator)