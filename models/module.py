from turtle import forward
import torch 
import torch.nn as nn
import numpy as np


class UNetModule(nn.Module):
    """
        (down_in) [down]                                     [up] (up_out)
            (down_out) |--------- inner_module --------- | (up_in)

    Args:
        nn (_type_): _description_
    """
    def __init__(self, inner_module:nn.Module, down_in_channels:int, down_out_channels:int, up_in_channels:int, up_out_channels:int, dropout:bool=False, batchnorm_down:bool=True, batchnorm_up:bool=True, relu:bool=True, last:bool=False) -> None:
        super().__init__()
        self.inner_module = inner_module
        self.down = Down(down_in_channels, down_out_channels, batchnorm=batchnorm_down, relu=relu)
        self.up = Up(up_in_channels, up_out_channels, dropout=dropout, batchnorm=batchnorm_up, bias=last == True)
        self.last = last
        
        if self.inner_module is not None:
            self.model = nn.Sequential(self.down, self.inner_module, self.up)
        
        else:
            self.model = nn.Sequential(self.down, self.up)

    def forward(self, x):
        print(x.shape, self.model(x).shape)
        if self.last: # no skip connection at the outermost layer
            return self.model(x)

        return torch.cat([x, self.model(x)], 1)


class Down(nn.Module):
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

