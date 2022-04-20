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
    def __init__(self, generator:nn.Module, test_loader:DataLoader, train_loader:DataLoader, learning_rate:float, betas:tuple) -> None:
        self.generator = generator
        self.L1_loss = nn.L1Loss()
        self.test_loader = test_loader
        self.train_loader = train_loader

        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)

    def plot_samples(self, file_name:str=None):
        multi_plot(self.test_loader, self.generator, file_name + ".png", columns=4)
    

class Pretrain(Trainer):
    def __init__(self, generator:nn.Module, test_loader:DataLoader, train_loader:DataLoader, learning_rate:float=0.0002, betas:tuple=(0.5, 0.999)) -> None:
        super().__init__(generator, test_loader, train_loader, learning_rate, betas)
        self.train_avg_loss = []
        self.test_avg_loss = []

    def train(self, nb_epochs:int, figures_path:str=None, start:int=0, generator_path:str=None, verbose:bool=True):
        self.generator.train()

        for epoch in range(start, nb_epochs):
            train_losses = []
            test_losses = []
            
            for L, ab in self.train_loader:
                L = L.to(device)
                ab = ab.to(device)
                pred = self.generator(L)
                loss = self.L1_loss(pred, ab)

                train_losses.append(loss.detach().to("cpu"))

                self.optimizer_G.zero_grad()
                loss.backward()
                self.optimizer_G.step()


            with torch.no_grad():   
                self.generator.eval()
                for L, ab in self.test_loader:
                    L = L.to(device)
                    ab = ab.to(device)

                    pred = self.generator(L)
                    loss = self.L1_loss(pred, ab)
                    test_losses.append(loss.detach().to("cpu"))

                self.train_avg_loss.append(torch.mean(torch.Tensor(train_losses)))
                self.test_avg_loss.append(torch.mean(torch.Tensor(test_losses)))

                if verbose:
                    print('[Epoch {}/{}] '.format(epoch+1, nb_epochs) +
                            'train_loss: {:.4f} - '.format(self.train_avg_loss[-1]) +
                            'test_loss: {:.4f}'.format(self.test_avg_loss[-1]))

                if figures_path is not None:
                    self.plot_samples(figures_path + "epoch_{}".format(epoch+1))
                
                if generator_path is not None:
                    torch.save(self.generator.state_dict(), generator_path + "generator")

    def make_plot(self, path:str):
        plt.figure(figsize=(16, 6))
        plt.title('Generator L1 loss')
        plt.plot(self.train_avg_loss)
        plt.plot(self.test_avg_loss)
        plt.grid()
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss (L1)')
        plt.savefig(path + "pretrain_Generator.png")


class GanTrain(Trainer):   
    def __init__(self, generator:nn.Module, discriminator:nn.Module, test_loader:DataLoader, train_loader:DataLoader, reg_R1:bool=False, learning_rate_g:float=0.0002, learning_rate_d:float=0.0002, betas_g:tuple=(0.5, 0.999), betas_d:tuple=(0.5, 0.999), gamma_1:float=100, gamma_2:float=1, real_label=1.0, fake_label=0.0) -> None:
        super().__init__(generator, test_loader, train_loader, learning_rate_g, betas_g)
        self.discriminator = discriminator
        self.cgan_loss = cGANLoss(real_label=real_label, fake_label=fake_label)

        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=betas_d)

        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.reg_R1 = reg_R1

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

        loss = L1_loss * self.gamma_1 + gan_loss

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
        torch.save(self.generator.state_dict(), path + "generator_{}".format(epoch+1))
        torch.save(self.discriminator.state_dict(), path + "discriminator_{}".format(epoch+1))


    def train(self, nb_epochs:int, models_path:str=None, logs_path:str=None, figures_path:str=None, start:int=0, verbose:bool=True, early_stopping:int=3):
        len_train = len(self.train_loader)
        len_test = len(self.test_loader)
        evalutation = Evalutation()

        # early stopping based on https://pythonguides.com/pytorch-early-stopping/
        n_epochs_stop = 6
        epochs_no_improve = 0
        early_stop = False
        min_val_loss = np.Inf
        best_model = None

        self.g_losses_avg = {"train": torch.zeros((nb_epochs, 3)), "val": torch.zeros((nb_epochs, 3))}
        self.d_losses_avg = {"train": torch.zeros((nb_epochs, 3)), "val": torch.zeros((nb_epochs, 3))}
        self.evaluation_avg = {"train": torch.zeros((nb_epochs, 2)), "val": torch.zeros(nb_epochs, 2)}

        for epoch in range(start, nb_epochs):
            evaluation = {"train": torch.zeros((len_train, 2)), "val": torch.zeros((len_test, 2))}
            g_losses_mem = {"train": torch.zeros((len_train, 3)), "val": torch.zeros((len_test, 3))}
            d_losses_mem = {"train": torch.zeros((len_train, 3)), "val": torch.zeros((len_test, 3))}
            
            for i, (L, ab) in enumerate(self.train_loader):
                L = L.to(device)
                ab = ab.to(device)
                fake_ab = self.generator(L)

                d_losses, g_losses = self._step(L, ab, fake_ab)
                d_losses_mem["train"][i] = torch.Tensor(d_losses)
                g_losses_mem["train"][i] = torch.Tensor(g_losses)

                with torch.no_grad():
                    evaluation["train"][i] = evalutation.eval(L.detach().to("cpu"), ab.detach().to("cpu"), fake_ab.detach().to("cpu"))


            with torch.no_grad():   
                # Do not set .eval()
                for i, (L, ab) in enumerate(self.test_loader):
                    L = L.to(device)
                    ab = ab.to(device)
                    fake_ab = self.generator(L)
                    
                    g_losses = self._generator_loss(L, ab, fake_ab, train=False)
                    d_losses = self._discriminator_loss(L, ab, fake_ab, train=False)

                    d_losses_mem["val"][i] = torch.Tensor(d_losses)
                    g_losses_mem["val"][i] = torch.Tensor(g_losses)
                    evaluation["val"][i] = evalutation.eval(L.detach().to("cpu"), ab.detach().to("cpu"), fake_ab.detach().to("cpu"))

                self.evaluation_avg["train"][epoch] = torch.mean(evaluation["train"], 0)
                self.evaluation_avg["val"][epoch] = torch.mean(evaluation["val"], 0)

                self.d_losses_avg["train"][epoch] = torch.mean(d_losses_mem["train"], 0)
                self.g_losses_avg["train"][epoch] = torch.mean(g_losses_mem["train"], 0)

                self.d_losses_avg["val"][epoch] = torch.mean(d_losses_mem["val"], 0)
                self.g_losses_avg["val"][epoch] = torch.mean(g_losses_mem["val"], 0)

                # === Early stopping === https://pythonguides.com/pytorch-early-stopping/
                if self.g_losses_avg["val"][epoch][1] < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = self.g_losses_avg["val"][epoch][1]
                    best_generator = copy.deepcopy(self.generator)

                else:
                    epochs_no_improve += 1
                    print('No improve {}/{}:'.format(epochs_no_improve, n_epochs_stop))

                if epochs_no_improve >= n_epochs_stop:
                    print('Early stopping!')
                    self.generator = best_generator
                    self.plot_samples(figures_path + "best")
                    self._save_models(models_path, epoch)
                    break
                # === End Early stopping ===

                if verbose:
                    print('[Epoch {}/{}] '.format(epoch+1, nb_epochs) + "\n--- Generator ---\n" +
                                '\tTrain: loss: {:.4f} - '.format(self.g_losses_avg["train"][epoch][0]) +'L1 loss: {:.4F} - '.format(self.g_losses_avg["train"][epoch][1]) +'cGan loss: {:.4F}'.format(self.g_losses_avg["train"][epoch][2]) +
                                '\n\tTest: loss: {:.4f} - '.format(self.g_losses_avg["val"][epoch][0]) +'L1 loss: {:.4F} - '.format(self.g_losses_avg["val"][epoch][1]) +'cGan loss: {:.4F}'.format(self.g_losses_avg["val"][epoch][2]) +       
                                "\n--- Discriminator ---\n" +
                                '\tTrain: loss: {:.4f} - '.format(self.d_losses_avg["train"][epoch][0]) + "R1: {:.4F} - ".format(self.d_losses_avg["train"][epoch][1]) + "cGan loss: {:.4F}".format(self.d_losses_avg["train"][epoch][2]) +
                                '\n\tTest: loss: {:.4f} - '.format(self.d_losses_avg["val"][epoch][0]) + "R1: {:.4F} - ".format(self.d_losses_avg["val"][epoch][1]) + "cGan loss: {:.4F}".format(self.d_losses_avg["val"][epoch][2]) +
                                '\n--- Metrics ---\n' + 
                                '\tTrain: SSIM: {:.4f} - PSNR: {:.4f}'.format(self.evaluation_avg["train"][epoch][0], self.evaluation_avg["train"][epoch][1]) + 
                                '\n\tTest: SSIM: {:.4f} - PSNR: {:.4f}'.format(self.evaluation_avg["val"][epoch][0], self.evaluation_avg["val"][epoch][1]))

                # if figures_path is not None:
            if (epoch+1) % 5 == 0:
                self.plot_samples(figures_path + "epoch_{}".format(epoch+1))
                self._save_models(models_path, epoch)

        torch.save(self.g_losses_avg["train"], logs_path + "g_losses_avg_train.pt")
        torch.save(self.g_losses_avg["val"], logs_path + "g_losses_avg_val.pt")
        torch.save(self.d_losses_avg["train"], logs_path + "d_losses_avg_train.pt")
        torch.save(self.d_losses_avg["val"], logs_path + "d_losses_avg_val.pt")
        torch.save(self.evaluation_avg["train"], logs_path + "evaluations_avg_train.pt")
        torch.save(self.evaluation_avg["val"], logs_path + "evaluations_avg_val.pt")

    # Generator losses
    # Discriminateur losses
    # GvsD train/val
    def make_plot(self, path:str):

        plt.figure(figsize=(16, 6))
        plt.title('Generator losses')
        plt.plot(self.train_g_avg_loss)
        plt.plot(self.test_g_avg_loss)
        plt.grid()
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path + "generator_losses.png")

        plt.figure(figsize=(16, 6))
        plt.title('Discriminator losses')
        plt.plot(self.train_d_avg_loss)
        plt.plot(self.test_d_avg_loss)
        plt.grid()
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path + "discriminator_losses.png")

        plt.figure(figsize=(16, 6))
        plt.title('Generator - cGan loss')
        plt.plot(self.train_gan_avg_loss)
        plt.plot(self.test_gan_avg_loss)
        plt.grid()
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path + "generator_cGan.png")

        plt.figure(figsize=(16, 6))
        plt.title('Generator vs Discriminator - cGan loss')
        plt.plot(self.test_gan_avg_loss)
        plt.plot(self.test_d_avg_loss)
        plt.grid()
        plt.legend(['Generator', 'Discriminator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path + "generator_vs_discriminator.png")

        plt.figure(figsize=(16, 6))
        plt.title('Generator val losses')
        plt.plot(self.test_l1_avg_loss)
        plt.plot(self.test_gan_avg_loss)
        plt.grid()
        plt.legend(['L1', 'cGan loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path + "Generator_testlosses.png")