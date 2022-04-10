import torch.nn as nn
import torch 

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
        self.model.apply(nn.init(nn.init.normal_(self.model.weight.data, 0.0, 0.02))) # init weights with a gaussian distribution centered at 0, and std=0.02

    def forward(self, x):
        return self.model(x)


class UNetModule(nn.Module):
    """
        A module of UNet

        (down_in) [down]                                 [up] (up_out)
            (down_out) |--------- inner_module --------- | (up_in)
    """
    def __init__(self, inner_module:nn.Module, down_in_channels:int, down_out_channels:int, up_in_channels:int, up_out_channels:int, dropout:bool=False, batchnorm_down:bool=True, batchnorm_up:bool=True, relu:bool=True, last:bool=False) -> None:
        super().__init__()
        self.inner_module = inner_module
        self.down = Down(down_in_channels, down_out_channels, batchnorm=batchnorm_down, relu=relu)
        self.up = Up(up_in_channels, up_out_channels, dropout=dropout, batchnorm=batchnorm_up, bias=last == True)
        self.last = last # outermost layer
        
        if self.inner_module is not None: 
            self.model = nn.Sequential(self.down, self.inner_module, self.up)
        
        else: # innermost (bottleneck 1x1 layer)
            self.model = nn.Sequential(self.down, self.up)

    def forward(self, x):
        if self.last: # no skip connection at the outermost layer
            return self.model(x)

        return torch.cat([x, self.model(x)], 1)


class Down(nn.Module):
    """
       Encoder block that downsample the input by a factor of 2
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=4, stride:int=2, padding:int=1, batchnorm:bool=True, relu:bool=True) -> None:
        super().__init__()
        down = []
        if relu:
            down += [nn.LeakyReLU(0.2, True)]

        down += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]

        if batchnorm:
            down += [nn.BatchNorm2d(out_channels)]

        self.down = nn.Sequential(*down)

    def forward(self, x:torch.Tensor):
        return self.down(x)


# https://discuss.pytorch.org/t/torch-nn-convtranspose2d-vs-torch-nn-upsample/30574
# Choisir entre ?
# Upsampling = interpolation
# ConvTranspose = trainable kernels
class Up(nn.Module):    
    """
       Decoder block that upsample the input by a factor of 2
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=4, stride:int=2, padding:int=1, dropout:bool=False, batchnorm:bool=True, bias:bool=True) -> None:
        super().__init__()
        up = [nn.ReLU(True), 
              nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]

        if batchnorm:
            up += [nn.BatchNorm2d(out_channels)]

        if dropout:
            up += [nn.Dropout2d(p=0.2)]

        self.up = nn.Sequential(*up)


    def forward(self, x:torch.Tensor):
        return self.up(x)
