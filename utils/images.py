import matplotlib.pyplot as plt 
import numpy as np
import torch

from skimage import  color
from torchvision import transforms
from PIL import Image

def tensor_to_pil(labs:torch.Tensor):
    """
       Transform tensors using lab colorspace to a RGB PIL image
    """
    images = []
    labs[:, [0]] = (labs[:, [0]] + 1)*50
    labs[:, [1, 2]] = labs[:, [1, 2]]*128
    
    for lab in labs:
        pil_lab = np.array(lab.permute(1, 2, 0))
        arr_lab = np.around((color.lab2rgb(pil_lab) * 255)).astype("uint8")
        images.append(Image.fromarray(arr_lab))
    return images

def show_images(img):
    plt.imshow(transforms.functional.to_pil_image(img))
    plt.show()