import torch 
from torch import nn 


class PatchGAN(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.model = []
        self.patch_size = 70
        #Blocks of the architecture
        self.C_block(channels, 64, False)
        self.C_block(64, 128)
        self.C_block(128, 256)
        self.C_block(256, 512)
        self.C_block(512, 1, False, False)
        
        
        #Activation function
        self.model += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*self.model)
        self.model.apply(nn.init(nn.init.normal_(self.model.weight.data, 0.0, 0.02))) # init weights with a gaussian distribution centered at 0, and std=0.02
    
    def C_block(self, in_channels, out_channels, batch=True, relu=True):
            self.model += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)]
            if batch:
                self.model += [nn.BatchNorm2d(out_channels)]
            if relu:
                self.model += [nn.LeakyReLU(0.2, True)]

    def forward(self, l, ab):
        x = torch.cat((l, ab), axis=1)
        return self.model(x)