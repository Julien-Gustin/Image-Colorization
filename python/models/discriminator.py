import torch 
from torch import nn 


class PatchGAN(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.model = []
        #Blocks of the architecture
        self.C_block(2*channels, 64, False)
        self.C_block(64, 128)
        self.C_block(128, 256)
        self.C_block(256, 512)
        self.C_block(512, 1, False, False)
        
        
        #Activation function
        self.model += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*self.model)
    
    def C_block(self, in_channels, out_channels, batch=True, relu=True):
            self.model += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)]
            if batch:
                self.model += [nn.BatchNorm2d(out_channels)]
            if relu:
                self.model += [nn.LeakyReLU(0.2, True)]

    def forward(self, src, target):
        """
        x must be a pair (src, target) where
        
        src.shape == target.shape 
        The number of channels of an image must be equal to the number provided when instantiating this class.
        """
        #If an image is of size 256x256x3, this operation provided an image of size 256x256x6
        x = torch.cat((src, target), axis=1)
        print(x.shape)
        return self.model(x)
        
        
        
        