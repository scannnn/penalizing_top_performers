import torch
import torch.nn as nn
import torch.nn.functional as F

class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        # FCN8s decoder
        # ReLU'da denenebilir
        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.in1     = nn.InstanceNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.in2     = nn.InstanceNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.in3     = nn.InstanceNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.in4     = nn.InstanceNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.in5     = nn.InstanceNorm2d(64)
        # out_channels here is class number
        self.classifier = nn.Conv2d(in_channels=64, out_channels=19, kernel_size=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        d1 = self.in1(self.leakyRelu(self.deconv1(x)))      # size=(N, 512, x.H/16, x.W/16)
        d2 = self.in2(self.leakyRelu(self.deconv2(d1)))  # size=(N, 256, x.H/8, x.W/8)
        d3 = self.in3(self.leakyRelu(self.deconv3(d2)))  # size=(N, 128, x.H/4, x.W/4)
        d4 = self.in4(self.leakyRelu(self.deconv4(d3)))  # size=(N, 64, x.H/2, x.W/2)
        d5 = self.in5(self.leakyRelu(self.deconv5(d4)))  # size=(N, 32, x.H, x.W)
        d6 = self.classifier(d5)                         # size=(N, n_class, x.H/1, x.W/1)

        return d6

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def build_generator():
    G = generator()
    normal_init(G, 0, 0.01)
    return G
