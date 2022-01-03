import torch
import torch.nn as nn
import torch.nn.functional as F


class generator(nn.Module):

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

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn2(out)
        out = self.relu(out)
        out = self.deconv1(out)

        d1 = self.in1(self.leakyRelu(self.deconv1(out)))
        d2 = self.in2(self.leakyRelu(self.deconv2(d1)))
        d3 = self.in3(self.leakyRelu(self.deconv3(d2)))           

        return d3
