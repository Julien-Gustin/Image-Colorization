# Image Colorization with GAN

In the early days of photography, the images captured by a camera were only in black and white. Given such a grayscale image, it is usually an easy task for a human to predict color of each object appearing in the picture with a strong degree of confidence. For example, if one recognizes grass on a certain area of a photo, one can predict the color of this area as green. Throughout history, many works have been done to manually add plausible colors to monochrome photographs. The main goal of this project is to automate such a procedure.\\

Given a grayscale image, we want to automatically colorize it without any user input. To achieve so, we will build and train a *conditional generative adversarial network* (cGAN) from scratch. Our neural network has to understand the different elements present on an image (segmentation) and recolor them in a coherent way for a human. We will start from an architecture that has already proven itself from existing works (cfr. [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf)). We will first understand the architecture in depth, then try variations in order to make it ours and  improve it. 

## Implementation & method

More information concerning the implementation and method on the [report](report.pdf).


## Reproductibility

1. Download the dataset ```./data/Coco/dl_data.sh```
2. Launch the training 
   - ```python3 train.py --epochs 6 --dataset "data/Coco" --version "2017" --folder_name "output" [--R1]```
   - \[--R1\] To enable the R1 regularization or not
   - Other parameters are available and explained in [train.py](train.py)

## Results

<p align="center">
  <img src="https://github.com/Julien-Gustin/Deep-Learning/blob/main/figures/sky.png?raw=true" />
    <img src="https://github.com/Julien-Gustin/Deep-Learning/blob/main/figures/car.png?raw=true" />
      <img src="https://github.com/Julien-Gustin/Deep-Learning/blob/main/figures/pretrain.png?raw=true" />
</p>