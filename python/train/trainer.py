import torch
import torch.nn as nn
import copy

from python.eval.evaluation import Evalutation
from python.utils.images import *
from .loss import cGANLoss, R1Loss
from torch.utils.data import DataLoader

import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://arxiv.org/pdf/1803.05400.pdf train + explication
# https://arxiv.org/pdf/1406.2661.pdf (train GAN)
class Trainer():
    def __init__(self, generator:nn.Module, val_loader:DataLoader, train_loader:DataLoader, learning_rate:float, betas:tuple) -> None:
        self.generator = generator
        self.L1_loss = nn.L1Loss()
        self.val_loader = val_loader
        self.train_loader = train_loader

        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)

    def plot_samples(self, file_name:str=None, noise:bool=False):
        multi_plot(self.val_loader, self.generator, file_name + ".png", columns=4, noise=noise)


class GanTrain(Trainer):   
    def __init__(self, generator:nn.Module, discriminator:nn.Module, val_loader:DataLoader, train_loader:DataLoader, reg_R1:bool=False, learning_rate_g:float=0.0002, learning_rate_d:float=0.0002, betas_g:tuple=(0.5, 0.999), betas_d:tuple=(0.5, 0.999), gamma_1:float=100, gamma_2:float=1, real_label=1.0, fake_label=0.0, gan_weight=1) -> None:
        super().__init__(generator, val_loader, train_loader, learning_rate_g, betas_g)
        self.discriminator = discriminator
        self.cgan_loss = cGANLoss(real_label=real_label, fake_label=fake_label)

        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=betas_d)
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.reg_R1 = reg_R1
        self.gan_weight = gan_weight

        if self.reg_R1:
            self.r1_loss = R1Loss(gamma=self.gamma_2)

    def _set_requires_grad(self, model:nn.Module, requires_grad:bool):
        for param in model.parameters():
            param.requires_grad = requires_grad 

    def _generator_loss(self, L:torch.Tensor, reel_ab:torch.Tensor, fake_ab, train:bool=True):
        """ Compute the loss of the generator according to a weighted sum of L1 loss and GAN loss """
        discriminator_confidence = self.discriminator(L, fake_ab)

        L1_loss = self.L1_loss(fake_ab, reel_ab)
        gan_loss = self.cgan_loss(discriminator_confidence, True) # trick

        loss = L1_loss * self.gamma_1 + gan_loss * self.gan_weight

        if not train:
            return loss.detach().to("cpu"), L1_loss, gan_loss.detach().to("cpu")

        return loss, L1_loss, gan_loss

    def _generator_step(self, L:torch.Tensor, reel_ab:torch.Tensor, fake_ab:torch.Tensor):
        """ Perform a one step generator training """
        loss, L1_loss, gan_loss = self._generator_loss(L, reel_ab, fake_ab)
        
        self.optimizer_G.zero_grad()
        loss.backward()
        self.optimizer_G.step()

        return loss.detach().to("cpu"), L1_loss.detach().to("cpu"), gan_loss.detach().to("cpu")

    def generate_fake_samples(self, L:torch.Tensor): # is it used?
        with torch.no_grad():
            fake_ab = self.generator(L)
        return fake_ab 
    
    def _discriminator_loss(self, L:torch.Tensor, real_ab:torch.Tensor, fake_ab:torch.Tensor, train:bool=True):
        #Compute the loss when samples are real images
        penalization = 0
        if self.reg_R1 and train:
            real_ab.requires_grad_()
            pred_D_real = self.discriminator(L, real_ab)
            penalization = self.r1_loss(pred_D_real, real_ab)

        else:
            pred_D_real = self.discriminator(L, real_ab)
        
        loss_over_real_img = self.cgan_loss(pred_D_real, True)
        
        #Compute the loss when samples are fake images
        pred_D_fake = self.discriminator(L, fake_ab)
        loss_over_fake_img = self.cgan_loss(pred_D_fake, False)

        gan_loss = (loss_over_real_img + loss_over_fake_img)/2

        if not train:
            return (gan_loss + penalization).detach().to("cpu"), penalization, gan_loss.detach().to("cpu")
        
        return gan_loss + penalization, penalization, gan_loss

    def _discriminator_step(self, L:torch.Tensor, real_ab:torch.Tensor, fake_ab:torch.Tensor):
        (loss, penalization, gan_loss) = self._discriminator_loss(L, real_ab, fake_ab)
        
        self.optimizer_D.zero_grad()
        loss.backward()
        self.optimizer_D.step()
        
        return loss.detach().to("cpu"), penalization, gan_loss.detach().to("cpu")
        
    def _step(self, L:torch.Tensor, reel_ab:torch.Tensor, fake_ab:torch.Tensor):
        #generate ab outputs from the generator
        self._set_requires_grad(self.discriminator, True)
        d_losses = self._discriminator_step(L, reel_ab, fake_ab.detach())

        self._set_requires_grad(self.discriminator, False)
        g_losses = self._generator_step(L, reel_ab, fake_ab)

        return d_losses, g_losses

    def _save_models(self, path, epoch):
        torch.save(self.generator.state_dict(), path + "generator_{}".format(epoch))
        torch.save(self.discriminator.state_dict(), path + "discriminator_{}".format(epoch))


    def train(self, nb_epochs:int, models_path:str=None, logs_path:str=None, figures_path:str=None, start:int=0, verbose:bool=True, early_stopping:int=3, noise:bool=False):
        len_train = len(self.train_loader)
        len_val = len(self.val_loader)
        print(len_train)
        evalutation = Evalutation()

        # early stopping based on https://pythonguides.com/pytorch-early-stopping/
        n_epochs_stop = early_stopping
        epochs_no_improve = 0
        # min_val_loss = np.Inf
        max_ssim = 0
        best_generator = None
        best_epoch = 0

        self.g_losses_avg = {"train": torch.zeros((nb_epochs, 3)), "val": torch.zeros((nb_epochs, 3))}
        self.d_losses_avg = {"train": torch.zeros((nb_epochs, 3)), "val": torch.zeros((nb_epochs, 3))}
        self.evaluation_avg = torch.zeros(nb_epochs, 2)

        self.generator.train()
        self.discriminator.train()

        for epoch in range(start, nb_epochs):
            g_losses_mem = {"train": torch.zeros((len_train, 3)), "val": torch.zeros((len_val, 3))}
            d_losses_mem = {"train": torch.zeros((len_train, 3)), "val": torch.zeros((len_val, 3))}
            
            for i, (L, ab) in enumerate(self.train_loader):
                L = L.to(device)
                ab = ab.to(device)

                if noise:
                    z = torch.randn(L.size()).to(device)
                    L_z = torch.cat((L, z), 1)
                    fake_ab = self.generator(L_z)

                else:
                    fake_ab = self.generator(L)

                d_losses, g_losses = self._step(L, ab, fake_ab)
                d_losses_mem["train"][i] = torch.Tensor(d_losses)
                g_losses_mem["train"][i] = torch.Tensor(g_losses)   

            with torch.no_grad():   
                evaluation_val = torch.zeros((len_val, 2))
                # Do not set .eval()
                for i, (L, ab) in enumerate(self.val_loader):
                    L = L.to(device)
                    ab = ab.to(device)

                    if noise:
                        z = torch.randn(L.size()).to(device)
                        L_z = torch.cat((L, z), 1)
                        fake_ab = self.generator(L_z)

                    else:
                        fake_ab = self.generator(L)
                    
                    g_losses = self._generator_loss(L, ab, fake_ab, train=False)
                    d_losses = self._discriminator_loss(L, ab, fake_ab, train=False)

                    d_losses_mem["val"][i] = torch.Tensor(d_losses)
                    g_losses_mem["val"][i] = torch.Tensor(g_losses)
                    evaluation_val[i] = evalutation.eval(L, ab, fake_ab)

                self.evaluation_avg[epoch] = torch.mean(evaluation_val, 0) 
                self.d_losses_avg["train"][epoch] = torch.mean(d_losses_mem["train"], 0)
                self.g_losses_avg["train"][epoch] = torch.mean(g_losses_mem["train"], 0)

                self.d_losses_avg["val"][epoch] = torch.mean(d_losses_mem["val"], 0)
                self.g_losses_avg["val"][epoch] = torch.mean(g_losses_mem["val"], 0)

                # === Early stopping ===

                if early_stopping != -1:

                    if self.evaluation_avg[epoch][0] > max_ssim:
                        epochs_no_improve = 0
                        max_ssim = self.evaluation_avg[epoch][0]
                        best_generator = copy.deepcopy(self.generator)
                        best_epoch = epoch

                    else:
                        epochs_no_improve += 1
                        print('No improve {}/{}:'.format(epochs_no_improve, n_epochs_stop))

                else:
                    best_generator = copy.deepcopy(self.generator)
                    best_epoch = epoch

                # === End Early stopping ===

                if verbose:
                    print('[Epoch {}/{}] '.format(epoch+1, nb_epochs) + "\n--- Generator ---\n" +
                                '\tTrain: loss: {:.4f} - '.format(self.g_losses_avg["train"][epoch][0]) +'L1 loss: {:.4F} - '.format(self.g_losses_avg["train"][epoch][1]) +'cGan loss: {:.4F}'.format(self.g_losses_avg["train"][epoch][2]) +
                                '\n\tVal: loss: {:.4f} - '.format(self.g_losses_avg["val"][epoch][0]) +'L1 loss: {:.4F} - '.format(self.g_losses_avg["val"][epoch][1]) +'cGan loss: {:.4F}'.format(self.g_losses_avg["val"][epoch][2]) +       
                                "\n--- Discriminator ---\n" +
                                '\tTrain: loss: {:.4f} - '.format(self.d_losses_avg["train"][epoch][0]) + "R1: {:.4F} - ".format(self.d_losses_avg["train"][epoch][1]) + "cGan loss: {:.4F}".format(self.d_losses_avg["train"][epoch][2]) +
                                '\n\tVal: loss: {:.4f} - '.format(self.d_losses_avg["val"][epoch][0]) + "R1: {:.4F} - ".format(self.d_losses_avg["val"][epoch][1]) + "cGan loss: {:.4F}".format(self.d_losses_avg["val"][epoch][2]) +
                                '\n--- Metrics ---\n' + 
                                '\tVal: SSIM: {:.4f} - PSNR: {:.4f}'.format(self.evaluation_avg[epoch][0], self.evaluation_avg[epoch][1]))

                if epochs_no_improve >= n_epochs_stop and early_stopping != -1:
                    print('Early stopping!')
                    break

            if (epoch+1) % 5 == 0:
                self.plot_samples(figures_path + "epoch_{}".format(epoch+1), noise=noise)
                self._save_models(models_path, epoch+1)

        self.generator = best_generator
        self.plot_samples(figures_path + "best_{}".format(best_epoch+1), noise=noise)
        self._save_models(models_path, "best_{}".format(best_epoch+1))
        torch.save(self.g_losses_avg["train"][:epoch+1], logs_path + "g_losses_avg_train.pt")
        torch.save(self.g_losses_avg["val"][:epoch+1], logs_path + "g_losses_avg_val.pt")
        torch.save(self.d_losses_avg["train"][:epoch+1], logs_path + "d_losses_avg_train.pt")
        torch.save(self.d_losses_avg["val"][:epoch+1], logs_path + "d_losses_avg_val.pt")
        torch.save(self.evaluation_avg[:epoch+1], logs_path + "evaluations_avg_val.pt")