import matplotlib.pyplot as plt 
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from skimage import  color
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_pil(labs:torch.Tensor):
    """
       Transform tensors using lab colorspace to a RGB PIL image
    """
    images = tensor_lab_to_rgb(labs)
    
    for i in range(len(images)):
        images[i] = Image.fromarray(images[i])

    return images

def tensor_lab_to_rgb(labs:torch.Tensor):
    images = []
    labs[:, [0]] = (labs[:, [0]] + 1)*50
    labs[:, [1, 2]] = labs[:, [1, 2]]*110
    
    for lab in labs:
        pil_lab = np.array(lab.permute(1, 2, 0))
        arr_lab = np.around((color.lab2rgb(pil_lab) * 255)).astype("uint8")
        images.append(arr_lab)

    return images

def show_images(img):
    plt.imshow(transforms.functional.to_pil_image(img))
    plt.show()

def multi_plot(loader, generator, file_name=None, rows=4, noise=False):
    L, real_ab = next(iter(loader))

    real_Lab = torch.concat((L, real_ab), 1)
    real_img = tensor_to_pil(torch.Tensor(real_Lab))

    gray_Lab = torch.concat((L, real_ab*0), 1)
    gray_img = tensor_to_pil(torch.Tensor(gray_Lab))

    if noise:
        z = torch.randn(L.size())
        L_z = torch.cat((L, z), 1)
        fake_ab = generator(L_z.to(device)).detach().to("cpu")

    else:
        fake_ab = generator(L.to(device)).detach().to("cpu")

    fake_Lab = torch.cat([L, fake_ab], axis=1)
    fake_img_L1 = tensor_to_pil(fake_Lab)

    imgs = real_img + gray_img + fake_img_L1

    plt.figure(figsize=(20,35))
    i = 1
    for j, _ in enumerate(real_img):

        ax = plt.subplot(rows+1, int(len(imgs) / rows + 1), i)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        # if j == 0:
        #     plt.title("Ground truth",  fontsize=20)

        plt.imshow(real_img[j])

        i += 1
        
        ax = plt.subplot(rows+1, int(len(imgs) / rows + 1), i)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(gray_img[j])
        # if j == 0:
        #     plt.title("Greyscale images", fontsize=20)
        
        i += 1

        ax = plt.subplot(rows+1, int(len(imgs) / rows + 1), i)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        # if j == 0:
        #     plt.title("Generated images", fontsize=20)
        plt.imshow(fake_img_L1[j])
        
        i += 1

    if file_name is not None:
        plt.savefig(file_name)