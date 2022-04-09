import matplotlib.pyplot as plt 
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from skimage import  color
from torchvision import transforms
from PIL import Image

def tensor_to_pil(labs:torch.Tensor):
    """
       Transform tensors using lab colorspace to a RGB PIL image
    """
    images = []
    labs[:, [0]] = (labs[:, [0]] + 1)*50
    labs[:, [1, 2]] = labs[:, [1, 2]]*110
    
    for lab in labs:
        pil_lab = np.array(lab.permute(1, 2, 0))
        arr_lab = np.around((color.lab2rgb(pil_lab) * 255)).astype("uint8")
        images.append(Image.fromarray(arr_lab))
    return images

def show_images(img):
    plt.imshow(transforms.functional.to_pil_image(img))
    plt.show()

def multi_plot(loader, generator, file_name, columns=5):
    L, real_ab = next(iter(loader))
    real_Lab = torch.concat((L, real_ab), 1)
    real_img = tensor_to_pil(torch.Tensor(real_Lab))


    gray_Lab = torch.concat((L, real_ab*0), 1)
    gray_img = tensor_to_pil(torch.Tensor(gray_Lab))

    fake_ab = generator(L).detach()
    fake_Lab = torch.cat([L, fake_ab], axis=1)
    fake_img_L1 = tensor_to_pil(fake_Lab)

    imgs = real_img + gray_img + fake_img_L1

    plt.figure(figsize=(30,20))
    for i, img in enumerate(imgs):
        # plt.tick_params(down = False)
        ax = plt.subplot(len(imgs) / columns + 1, columns, i + 1)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        
    plt.savefig(file_name)