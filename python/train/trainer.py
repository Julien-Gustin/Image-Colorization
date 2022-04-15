import torch
import torch.nn as nn

from python.utils.images import *
from .loss import cGANLoss, R1Loss
from torch.utils.data import DataLoader

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
        multi_plot(self.test_loader, self.generator, "figures/{}.png".format(file_name), columns=4)
    

class Pretrain(Trainer):
    def __init__(self, generator:nn.Module, test_loader:DataLoader, train_loader:DataLoader, learning_rate:float=0.0002, betas:tuple=(0.5, 0.999)) -> None:
        super().__init__(generator, test_loader, train_loader, learning_rate, betas)
        self.train_avg_loss = []
        self.test_avg_loss = []

    def train(self, nb_epochs:int, file_name_plot:str=None, start:int=0, generator_file:str=None, verbose:bool=True):
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

                if file_name_plot is not None:
                    self.plot_samples(file_name_plot + "_epoch_{}".format(epoch+1))
                
                if generator_file is not None:
                    torch.save(self.generator.state_dict(), "saved_models/{}".format(generator_file))

    def make_plot(self, prefix:str):
        plt.figure(figsize=(16, 6))
        plt.title('Generator L1 loss')
        plt.plot(self.train_avg_loss)
        plt.plot(self.test_avg_loss)
        plt.grid()
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss (L1)')
        plt.savefig("figures/{}_pretrain_Generator.png".format(prefix))


class GanTrain(Trainer):   
    def __init__(self, generator:nn.Module, discriminator:nn.Module, test_loader:DataLoader, train_loader:DataLoader, reg_R1:bool=False, learning_rate_g:float=0.0002, learning_rate_d:float=0.0002, betas_g:tuple=(0.5, 0.999), betas_d:tuple=(0.5, 0.999), gamma_1:float=100, gamma_2:float=1) -> None:
        super().__init__(generator, test_loader, train_loader, learning_rate_g, betas_g)
        self.discriminator = discriminator
        self.cgan_loss = cGANLoss()

        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_d, betas=betas_d)

        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.reg_R1 = reg_R1

        self.train_g_avg_loss = []
        self.train_d_avg_loss = []
        self.test_g_avg_loss = []
        self.test_d_avg_loss = []
        self.train_l1_avg_loss = []
        self.test_l1_avg_loss = []
        self.train_gan_avg_loss = []
        self.test_gan_avg_loss = []

        if self.reg_R1:
            self.r1_loss = R1Loss(gamma=self.gamma_2)

    def _set_requires_grad(self, model:nn.Module, requires_grad:bool):
        for param in model.parameters():
            param.requires_grad = requires_grad 

    def _generator_loss(self, L:torch.Tensor, reel_ab:torch.Tensor, fake_ab):
        """ Compute the loss of the generator according to a weighted sum of L1 loss and GAN loss """
        discriminator_confidence = self.discriminator(L, fake_ab)

        L1_loss = self.L1_loss(fake_ab, reel_ab)
        gan_loss = self.cgan_loss(discriminator_confidence, True) # trick

        loss = L1_loss * self.gamma_1 + gan_loss

        return loss, L1_loss, gan_loss

    def _generator_step(self, L:torch.Tensor, reel_ab:torch.Tensor, fake_ab:torch.Tensor):
        """ Perform a one step generator training """
        loss, L1_loss, gan_loss = self._generator_loss(L, reel_ab, fake_ab)
        
        self.optimizer_G.zero_grad()
        loss.backward()
        self.optimizer_G.step()

        return loss, L1_loss, gan_loss

    def generate_fake_samples(self, L:torch.Tensor): # is it used?
        with torch.no_grad():
            fake_ab = self.generator(L)
        return fake_ab 
    
    def _discriminator_loss(self, L:torch.Tensor, real_ab:torch.Tensor, fake_ab:torch.Tensor, train:bool=True):
        #Compute the loss when samples are real images
        loss_over_real_img = 0
        if self.reg_R1 and train:
            real_ab.requires_grad_()
            pred_D_real = self.discriminator(L, real_ab)
            loss_over_real_img += self.r1_loss(pred_D_real, real_ab)

        else:
            pred_D_real = self.discriminator(L, real_ab)
        
        loss_over_real_img += self.cgan_loss(pred_D_real, True)
        
        #Compute the loss when samples are fake images
        pred_D_fake = self.discriminator(L, fake_ab)
        loss_over_fake_img = self.cgan_loss(pred_D_fake, False)
        
        loss = (loss_over_real_img + loss_over_fake_img)/2
        
        return loss

    def _discriminator_step(self, L:torch.Tensor, real_ab:torch.Tensor, fake_ab:torch.Tensor):
        loss = self._discriminator_loss(L, real_ab, fake_ab)
        
        self.optimizer_D.zero_grad()
        loss.backward()
        self.optimizer_D.step()
        
        return loss 
        
    def _step(self, L:nn.Module, reel_ab:nn.Module):
        #generate ab outputs from the generator
        fake_ab = self.generator(L)

        self._set_requires_grad(self.discriminator, True)
        d_loss = self._discriminator_step(L, reel_ab, fake_ab.detach())

        self._set_requires_grad(self.discriminator, False)
        g_loss, L1_loss, gan_loss = self._generator_step(L, reel_ab, fake_ab)

        return d_loss.detach(), (g_loss.detach(), L1_loss.detach(), gan_loss.detach())


    def train(self, nb_epochs:int, generator_file:str=None, discriminator_file:str=None, file_name_plot:str=None, start:int=0, verbose:bool=True):
        for epoch in range(start, nb_epochs):
            train_g_losses = []
            train_d_losses = []
            test_g_losses = []
            test_d_losses = []
            train_L1_loss = []
            test_L1_loss = []
            train_gan_loss = []
            test_gan_loss = []
            
            for L, ab in self.train_loader:
                L = L.to(device)
                ab = ab.to(device)

                d_loss, (g_loss, L1_loss, gan_loss) = self._step(L, ab)

                train_g_losses.append(g_loss.detach().to("cpu"))
                train_d_losses.append(d_loss.detach().to("cpu"))  
                train_L1_loss.append(L1_loss.detach().to("cpu"))
                train_gan_loss.append(gan_loss.detach().to("cpu"))

            with torch.no_grad():   
                # Do not set .eval()
                for L, ab in self.test_loader:
                    L = L.to(device)
                    ab = ab.to(device)
                    fake_ab = self.generator(L)
                    (g_loss, L1_loss, gan_loss) = self._generator_loss(L, ab, fake_ab)
                    d_loss = self._discriminator_loss(L, ab, fake_ab, train=False)
                    
                    test_g_losses.append(g_loss.detach().to("cpu"))
                    test_d_losses.append(d_loss.detach().to("cpu"))
                    test_L1_loss.append(L1_loss.detach().to("cpu"))
                    test_gan_loss.append(gan_loss.detach().to("cpu"))

                self.train_g_avg_loss.append(torch.mean(torch.Tensor(train_g_losses)).to("cpu"))
                self.train_d_avg_loss.append(torch.mean(torch.Tensor(train_d_losses)).to("cpu"))
                self.test_g_avg_loss.append(torch.mean(torch.Tensor(test_g_losses)).to("cpu"))
                self.test_d_avg_loss.append(torch.mean(torch.Tensor(test_d_losses)).to("cpu"))
                self.train_l1_avg_loss.append(torch.mean(torch.Tensor(train_L1_loss)).to("cpu")) 
                self.test_l1_avg_loss.append(torch.mean(torch.Tensor(test_L1_loss)).to("cpu"))
                self.train_gan_avg_loss.append(torch.mean(torch.Tensor(train_gan_loss)).to("cpu"))
                self.test_gan_avg_loss.append(torch.mean(torch.Tensor(test_gan_loss)).to("cpu"))

                if verbose:
                    print('[Epoch {}/{}] '.format(epoch+1, nb_epochs) + "\n--- Generator ---\n" +
                                '\tTrain: loss: {:.4f} - '.format(self.train_g_avg_loss[-1]) +'L1 loss: {:.4F} - '.format(self.train_l1_avg_loss[-1]) +'cGan loss: {:.4F}'.format(self.train_gan_avg_loss[-1]) +
                                '\n\tTest: loss: {:.4f} - '.format(self.test_g_avg_loss[-1]) +'L1 loss: {:.4F} - '.format(self.test_l1_avg_loss[-1]) +'cGan loss: {:.4F}'.format(self.test_gan_avg_loss[-1]) +       
                                "\n--- Discriminator ---\n" +
                                '\ttrain_loss: {:.4f} - '.format(self.train_d_avg_loss[-1]) +
                                'test_loss: {:.4f}'.format(self.test_d_avg_loss[-1]))

                if file_name_plot is not None:
                    self.plot_samples(file_name_plot + "_epoch_{}".format(epoch+1))
                    
                if generator_file is not None:
                    torch.save(self.generator.state_dict(), "saved_models/{}".format(generator_file))

                if discriminator_file is not None:
                    torch.save(self.discriminator.state_dict(), "saved_models/{}".format(discriminator_file))

    def make_plot(self, prefix:str):
        plt.figure(figsize=(16, 6))
        plt.title('Generator losses')
        plt.plot(self.train_g_avg_loss)
        plt.plot(self.test_g_avg_loss)
        plt.grid()
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("figures/{}_generator_losses.png".format(prefix))

        plt.figure(figsize=(16, 6))
        plt.title('Discriminator losses')
        plt.plot(self.train_d_avg_loss)
        plt.plot(self.test_d_avg_loss)
        plt.grid()
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("figures/{}_discriminator_losses.png".format(prefix))

        plt.figure(figsize=(16, 6))
        plt.title('Generator - cGan loss')
        plt.plot(self.train_gan_avg_loss)
        plt.plot(self.test_gan_avg_loss)
        plt.grid()
        plt.legend(['Train', 'Test'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("figures/{}_generator_cGan.png".format(prefix))

        plt.figure(figsize=(16, 6))
        plt.title('Generator vs Discriminator - cGan loss')
        plt.plot(self.test_gan_avg_loss)
        plt.plot(self.test_d_avg_loss)
        plt.grid()
        plt.legend(['Generator', 'Discriminator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("figures/{}_generator_vs_discriminator.png".format(prefix))

        plt.figure(figsize=(16, 6))
        plt.title('Generator test losses')
        plt.plot(self.test_l1_avg_loss)
        plt.plot(self.test_gan_avg_loss)
        plt.grid()
        plt.legend(['L1', 'cGan loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("figures/{}_Generator_testlosses.png".format(prefix))