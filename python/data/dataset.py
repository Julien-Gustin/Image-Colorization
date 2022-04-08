import torch.utils.data as data
import os
import numpy as np

from skimage import color
from PIL import Image, ImageOps
from torchvision import transforms
from glob import glob

class CocoLab(data.Dataset):
    """Coco dataset using L*a*b colorspace"""
    def __init__(self, root_dir:str, version:str="2017", size:int=256, train:bool=True):
        """Initializes a dataset containing colored images."""
        super().__init__()
        self.train = train

        if self.train: # data augmentation?
            self.transform = transforms.Compose([transforms.Resize((size, size)), transforms.RandomHorizontalFlip()])

        else:
            self.transform = transforms.Compose([transforms.Resize((size, size))])

        folder_name = "train{}".format(version) if train else "val{}".format(version)
        self.data_paths = glob(os.path.join(root_dir, folder_name, "*.jpg"))

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data_paths)

    
    def __getitem__(self, index):
        """Returns the index-th data item of the dataset.

        Returns: #https://en.wikipedia.org/wiki/CIELAB_color_space,
            x(torch.Tensor): Perceptual lightness (L*) of the image, values between [0, 100]
            y(torch.Tensor):(a*b*) of the image, a - theorically unbounded 
                                                 b - theorically unbounded
        """
        path_i = self.data_paths[index]

        im = Image.open(path_i)
        if im.mode != "RGB":
            im = im.convert('RGB')

        rgb_im = self.transform(im)
        # could be interesting to use network in double instead
        lab_im = transforms.ToTensor()(color.rgb2lab(rgb_im).astype(np.float32)) # transform
        
        # Should we normalize between [-1, 1]
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/data/colorization_dataset.py
        # Dans le source code divise par 110, mais a,b sont unbounded et souvent on bound à [-127, 128], peut etre compute par rapport à RGB?
        # Solution ? https://fairyonice.github.io/Color-space-defenitions-in-python-RGB-and-LAB.html [-128, 128]
        L = lab_im[0:1, :, :] / 50 - 1
        ab = lab_im[1:3,:, :] / 110  #[-127, 128]
        
        return L, ab

class CocoGrayscaleRGB(data.Dataset):
    """Coco dataset using RGB colorspace"""
    def __init__(self, root_dir, size, train=True):
        """Initializes a dataset containing colored images."""
        super().__init__()
        self.train = train

        if self.train: # data augmentation?
            self.transform = transforms.Compose([transforms.Resize((size, size))])

        else:
            self.transform = transforms.Compose([transforms.Resize((size, size))])

        folder_name = "train2017" if train else "val2017"
        self.data_paths = glob(os.path.join(root_dir, folder_name, "*.jpg"))

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data_paths)


    def __getitem__(self, index):
        """Returns the index-th data item of the dataset.

        Returns:
            x(torch.Tensor) : Gray scale image, values between [0, 1]
            y(torch.Tensor): RGB image, values between [0, 1]
        """
        path_i = self.data_paths[index]

        im = Image.open(path_i)
        if im.mode != "RGB":
            im = im.convert('RGB')

        rgb_im = self.transform(im)

        # between [0, 1]
        gray_scale = transforms.ToTensor()(ImageOps.grayscale(rgb_im))
        rgb_im = transforms.ToTensor()(rgb_im)
        
        return gray_scale, rgb_im
