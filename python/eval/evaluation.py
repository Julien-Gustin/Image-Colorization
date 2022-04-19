from tabnanny import verbose
from piqa import PSNR, SSIM, LPIPS # https://github.com/francois-rozet/piqa
from python.utils.images import *

import torch

class Evalutation():
    def __init__(self, verbose:bool=True) -> None:
        self.metrics = []   
        self.verbose = verbose

        self.SSIM = SSIM(value_range=256)
        self.PSNR = PSNR(value_range=256)

    
    def eval(self, L, ab_pred, ab_target):
        real_Lab = torch.concat((L, ab_target), 1)
        real_RGB = torch.Tensor(tensor_lab_to_rgb(torch.Tensor(real_Lab))).permute(0, 3, 1, 2)

        fake_Lab = torch.cat([L, ab_pred], 1)
        fake_RGB = torch.Tensor(tensor_lab_to_rgb(fake_Lab)).permute(0, 3, 1, 2)
        with torch.no_grad():
            ssim = self.SSIM(fake_RGB, real_RGB)
            psnr = self.PSNR(fake_RGB, real_RGB)
        
        if verbose:
            print("==== Evaluation ====")

            for metric_name, metric in zip(["ssim", "psnr"], [ssim, psnr]):
                print("  {}: {:.4f}".format(metric_name, metric))
                print("-"*20)

        return torch.tensor([ssim, psnr])
