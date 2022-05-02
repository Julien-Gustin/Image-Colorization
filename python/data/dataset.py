import os
import numpy as np
import torch.utils.data as data

from skimage import color
from PIL import Image, ImageOps
from torchvision import transforms
from glob import glob

class CocoLab(data.Dataset):
    """Coco dataset using L*a*b colorspace"""
    def __init__(self, root_dir:str, splits, version:str="2017", size:int=256, train:bool=False):
        """Initializes a dataset containing colored images."""
        super().__init__()

        if train: # data augmentation
            self.transform = transforms.Compose([transforms.Resize((size, size)), transforms.RandomHorizontalFlip()])

        else:
            self.transform = transforms.Compose([transforms.Resize((size, size))])

        if isinstance(splits, list):
            self.data_paths = []
            for split in splits:
                folder_name = "{}{}".format(split, version)
                self.data_paths += glob(os.path.join(root_dir, folder_name, "*.jpg"))

        else:
            split = splits
            folder_name = "{}{}".format(split, version)
            self.data_paths = glob(os.path.join(root_dir, folder_name, "*.jpg"))

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data_paths)

    
    def __getitem__(self, index):
        """Returns the index-th data item of the dataset.

        Returns: #https://en.wikipedia.org/wiki/CIELAB_color_space,
            x(torch.Tensor): Perceptual lightness (L*) of the image, values between [0, 100]
            y(torch.Tensor):(a*b*) of the image, a - theorically unbounded (not in practice)
                                                 b - theorically unbounded (not in practice)
        """
        path_i = self.data_paths[index]

        im = Image.open(path_i)
        if im.mode != "RGB":
            im = im.convert('RGB')

        rgb_im = self.transform(im)
        lab_im = transforms.ToTensor()(color.rgb2lab(rgb_im).astype(np.float32)) # transform
        
        # Normalize between [-1, 1]
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/data/colorization_dataset.py
        # Divide by 110 in the source code, why? while most of the time they are bound to [-127, 128], might be computed with RGB?
        # Solution => https://stackoverflow.com/questions/19099063/what-are-the-ranges-of-coordinates-in-the-cielab-color-space
        L = lab_im[0:1, :, :] / 50 - 1
        ab = lab_im[1:3,:, :] / 110 # see ligne 58
        
        return L, ab
