from __future__ import absolute_import, division, print_function
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

def vgg_16(batch_norm=True, pretrained=False, fixed_feature=True):
    model = vgg16(pretrained=True)
    #model.load_state_dict(torch.load("/Users/can.cetindag/Documents/PERSONAL/AI/PROJECT/penalizing_top_performers/models/vgg16-397923af.pth"))
    return _VGG(model, model.features, True)

class FCN8(torch.nn.Module):

    def __init__(self, n_classes, pretrained_model: SqueezeExtractor):
        super(FCN8, self).__init__()
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

        o = x
        for i in range(len(self.features)):
            o = self.features[i](o)
            if i == self.copy_feature_info[-3].index or\
                    i == self.copy_feature_info[-2].index:
                saved_pools.append(o)

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

        return o


def fcn8_vgg16(n_classes):
    vgg = vgg_16()
    return FCN8(n_classes, vgg)
