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

def multi_plot(loader, generator, file_name=None, columns=4, noise=False):
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

    plt.figure(figsize=(30,15))
    for i, img in enumerate(imgs):
        ax = plt.subplot(int(len(imgs) / columns + 1), columns, i + 1)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(img)

    if file_name is not None:
        plt.savefig(file_name)

def multi_plot_rows(loader, generator, file_name=None, rows=4, noise=False, display_title=True, only_one=False):
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


    plt.figure(figsize=(20,35))
    i = 1

    for j, _ in enumerate(real_img):

        ax = plt.subplot(rows+1, 3, i)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if j == 0 and display_title:
            plt.title("Ground truth",  fontsize=20)

        plt.imshow(real_img[j])

        i += 1
        
        ax = plt.subplot(rows+1, 3, i)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if j == 0 and display_title:
            plt.title("Greyscale images", fontsize=20)

        plt.imshow(gray_img[j])
        
        i += 1

        ax = plt.subplot(rows+1, 3, i)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if j == 0 and display_title:
            plt.title("Generated images", fontsize=20)

        plt.imshow(fake_img_L1[j])

        if only_one:
            break
        
        i += 1

    if file_name is not None:
        plt.savefig(file_name)

def multi_plot_generators(loader, generators, labels, pretrain=None):
    L, real_ab = next(iter(loader))

    real_Lab = torch.concat((L, real_ab), 1)
    real_img = tensor_to_pil(torch.Tensor(real_Lab))

    gray_Lab = torch.concat((L, real_ab*0), 1)
    gray_img = tensor_to_pil(torch.Tensor(gray_Lab))

    fig, axs = plt.subplots(len(L), len(labels)+2, figsize=((len(labels)+2)*3, len(L)*3))
    # fig.suptitle('Sharing x per column, y per row')

    fake_imgs = []

    for generator in generators:
        fake_ab = generator(L.to(device)).detach().to("cpu")
        fake_Lab = torch.cat([L, fake_ab], axis=1)
        fake_imgs.append(tensor_to_pil(fake_Lab))

    if pretrain is not None:
        z = torch.randn(L.size())
        L_z = torch.cat((L, z), 1)
        fake_ab = pretrain(L_z.to(device)).detach().to("cpu")
        fake_Lab = torch.cat([L, fake_ab], axis=1)
        fake_imgs.append(tensor_to_pil(fake_Lab))


    axs[0][0].set_title("Ground truth",  fontsize=20)
    axs[0][1].set_title("Greyscale",  fontsize=20)
    for j in range(len(labels)):
        axs[0][2+j].set_title(labels[j],  fontsize=20)

    for i in range(len(axs)):
        axs[i][0].imshow(real_img[i])
        axs[i][1].imshow(gray_img[i])
        axs[i][0].axes.xaxis.set_visible(False)
        axs[i][0].axes.yaxis.set_visible(False)
        axs[i][1].axes.xaxis.set_visible(False)
        axs[i][1].axes.yaxis.set_visible(False)
        for j in range(len(labels)):
            axs[i][2+j].imshow(fake_imgs[j][i])
            axs[i][2+j].axes.xaxis.set_visible(False)
            axs[i][2+j].axes.yaxis.set_visible(False)


def plot_turing(loader, generator, file_name=None, rows=1, fake=False):
    L, real_ab = next(iter(loader))

    real_Lab = torch.concat((L, real_ab), 1)
    real_img = tensor_to_pil(torch.Tensor(real_Lab))

    gray_Lab = torch.concat((L, real_ab*0), 1)
    gray_img = tensor_to_pil(torch.Tensor(gray_Lab))

    fake_ab = generator(L.to(device)).detach().to("cpu")

    fake_Lab = torch.cat([L, fake_ab], axis=1)
    fake_img_L1 = tensor_to_pil(fake_Lab)

    plt.figure(figsize=(20,35))

    i = 1
    for j, _ in enumerate(real_img):
        ax = plt.subplot(rows+1, 2, i)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(gray_img[j])
        
        i += 1

        if fake:
            ax = plt.subplot(rows+1, 2, i)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.imshow(fake_img_L1[j])


        else:
            ax = plt.subplot(rows+1, 2, i)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.imshow(real_img[j])
        
        i += 1

    if file_name is not None:
        plt.savefig(file_name)

def plot_turing2(loader, generator, file_name=None, rows=1, fake=False):
    L, real_ab = next(iter(loader))

    real_Lab = torch.concat((L, real_ab), 1)
    real_img = tensor_to_pil(torch.Tensor(real_Lab))

    gray_Lab = torch.concat((L, real_ab*0), 1)
    gray_img = tensor_to_pil(torch.Tensor(gray_Lab))

    z = torch.randn(L.size())
    L_z = torch.cat((L, z), 1)

    fake_ab = generator(L_z.to(device)).detach().to("cpu")

    fake_Lab = torch.cat([L, fake_ab], axis=1)
    fake_img_L1 = tensor_to_pil(fake_Lab)

    plt.figure(figsize=(20,35))

    i = 1
    for j, _ in enumerate(real_img):
        ax = plt.subplot(rows+1, 2, i)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(gray_img[j])
        
        i += 1

        if fake:
            ax = plt.subplot(rows+1, 2, i)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.imshow(fake_img_L1[j])


        else:
            ax = plt.subplot(rows+1, 2, i)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.imshow(real_img[j])
        
        i += 1

    if file_name is not None:
        plt.savefig(file_name)