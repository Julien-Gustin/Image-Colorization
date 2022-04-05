import torch.nn as nn

from models.module import *

# Useful links:

#https://arxiv.org/pdf/1502.03167.pdf
#https://arxiv.org/pdf/1505.04597.pdf
#https://arxiv.org/pdf/1611.07004.pdf
#https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/003efc4c8819de47ff11b5a0af7ba09aee7f5fc1/models/networks.py#L436
#https://madebyollin.github.io/convnet-calculator/ - Using kernel=4, stride=2, padding=1 allow to divide by two at each layer

class UNet(nn.Module):
    """
       Implementation of the UNet architecture inspired from https://arxiv.org/pdf/1611.07004.pdf
    """
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        inner_module = UNetModule(None, 512, 512, 512, 512, batchnorm_down=False) # 1x1 -> 2x2 - bottleneck
        inner_module = UNetModule(inner_module, 512, 512, 512+512, 512, dropout=True) # 2x2 -> 4x4
        inner_module = UNetModule(inner_module, 512, 512, 512+512, 512, dropout=True) # 4x4 -> 8x8
        inner_module = UNetModule(inner_module, 512, 512, 512+512, 512, dropout=True) # 8x8 -> 16x16
        inner_module = UNetModule(inner_module, 256, 512, 512+512, 256) # 16x16 -> 32x32
        inner_module = UNetModule(inner_module, 128, 256, 256+256, 128) # 32x32 -> 64x64
        inner_module = UNetModule(inner_module, 64, 128, 128+128, 64) # 64x64 -> 128x128

        last = UNetModule(inner_module, self.in_channels, 64, 64+64, 2, relu=False, batchnorm_down=False, batchnorm_up=False, last=True) # 128x128 -> 256x256
        self.model = nn.Sequential(last, nn.Tanh()) # 256x256

    def forward(self, x):
        return self.model(x)