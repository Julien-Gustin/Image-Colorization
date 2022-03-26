import torch.utils.data as data
import os

from skimage import color
from PIL import Image, ImageOps
from torchvision import transforms
from glob import glob

class CocoLab(data.Dataset):
    """Coco dataset using L*a*b colorspace"""
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
            x(torch.Tensor): Perceptual lightness (L*) of the image
            y(torch.Tensor):(a*b*) of the image
        """
        path_i = self.data_paths[index]

        im = Image.open(path_i)
        if im.mode != "RGB":
            im = im.convert('RGB')
        
        rgb_im = self.transform(im)
        lab_im = transforms.ToTensor()(color.rgb2lab(rgb_im)) # transform
        
        # Should we normalize between [-1, 1] ? How? https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
        L = lab_im[0:1, :, :]
        ab = lab_im[1:3,:, :] 
        
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
