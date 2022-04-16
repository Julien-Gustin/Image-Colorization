from python.piqaa.piqaa import PSNR, SSIM, LPIPS # https://github.com/francois-rozet/piqa
from python.utils.images import *

import torch

class Evalutation():
    def __init__(self, psnr:bool=True, ssim:bool=True) -> None:
        self.metrics = []

        if ssim:
            SSIM_ = SSIM(value_range=256)
            self.metrics.append((SSIM_, "ssim"))

        if psnr:
            PSNR_ = PSNR(value_range=256)
            self.metrics.append((PSNR_, "psnr"))

    
    def eval(self, L, ab_pred, ab_target):
        real_Lab = torch.concat((L, ab_target), 1)
        real_RGB = torch.Tensor(tensor_lab_to_rgb(torch.Tensor(real_Lab))).permute(0, 3, 1, 2)

        fake_Lab = torch.cat([L, ab_pred], 1)
        fake_RGB = torch.Tensor(tensor_lab_to_rgb(fake_Lab)).permute(0, 3, 1, 2)

        print("==== Evaluation ====")

        for metric, metric_name in self.metrics:
            print("  {}: {:.4f}".format(metric_name, metric(fake_RGB, real_RGB)))
            print("-"*20)