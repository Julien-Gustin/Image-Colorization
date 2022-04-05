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
    for lab in labs:
        
        lab[0] = (lab[0] + 1)*50
        for i in [1,2]:
            lab[i] = lab[i]*128
        

        pil_lab = np.array(lab.permute(1, 2, 0))
        arr_lab = np.around((color.lab2rgb(pil_lab) * 255)).astype("uint8")
        images.append(Image.fromarray(arr_lab))
    return images

def show_images(img):
    plt.imshow(transforms.functional.to_pil_image(img))
    plt.show()