from matplotlib.pyplot import axis
import torch
import torch.nn as nn

from .loss import cGANLoss, R1Loss

# https://arxiv.org/pdf/1803.05400.pdf train + explication
# https://arxiv.org/pdf/1406.2661.pdf (train GAN)
class GanTrain():   
    def __init__(self, generator:nn.Module, discriminator:nn.Module, reg_R1:bool=False) -> None:
        self.generator = generator
        self.discriminator = discriminator

        self.cgan_loss = cGANLoss()
        self.l1_loss = nn.L1Loss()

        # TODO: hyperparameters in arguments
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.gamma = 100
        self.reg_R1 = reg_R1
        #self.r1_loss = R1Loss(gamma=1)

    def set_requires_grad(self, model:nn.Module, requires_grad:bool):
        for param in model.parameters():
            param.requires_grad = requires_grad 

    def generator_loss(self, L:torch.Tensor, reel_ab:torch.Tensor, fake_ab):
        """ Compute the loss of the generator according to a weighted sum of L1 loss and GAN loss """
        discriminator_confidence = self.discriminator(L, fake_ab)

        l1_loss = self.l1_loss(fake_ab, reel_ab)
        gan_loss = self.cgan_loss(discriminator_confidence, True) # trick

        loss = l1_loss * self.gamma + gan_loss

        return loss, l1_loss, gan_loss

    def generator_step(self, L:torch.Tensor, reel_ab:torch.Tensor, fake_ab:torch.Tensor):
        """ Perform a one step generator training """
        loss, l1_loss, gan_loss = self.generator_loss(L, reel_ab, fake_ab)
        
        self.optimizer_G.zero_grad()
        loss.backward()
        self.optimizer_G.step()

        return loss, l1_loss, gan_loss

    def generate_fake_samples(self, L:torch.Tensor):
        with torch.no_grad():
            fake_ab = self.generator(L)
        return fake_ab 
    
    def discriminator_loss(self, L:torch.Tensor, real_ab:torch.Tensor, fake_ab:torch.Tensor, train:bool=True):
        #Compute the loss when samples are real images
        loss_over_real_img = 0
        if self.reg_R1 and train:
            
            real_ab.requires_grad_()
            pred_D_real = self.discriminator(L, real_ab)
            R1 = R1Loss(gamma=1) 
            loss_over_real_img += R1(pred_D_real, real_ab)
            print("+++ R1 loss in training :", loss_over_real_img)

        else:
            pred_D_real = self.discriminator(L, real_ab)
        
        loss_over_real_img += self.cgan_loss(pred_D_real, True)
        
        #Compute the loss when samples are fake images
        pred_D_fake = self.discriminator(L, fake_ab)
        loss_over_fake_img = self.cgan_loss(pred_D_fake, False)
        
        loss = (loss_over_real_img + loss_over_fake_img)/2
        
        return loss

    def discriminator_step(self, L:torch.Tensor, real_ab:torch.Tensor, fake_ab:torch.Tensor):
        loss = self.discriminator_loss(L, real_ab, fake_ab)
        
        self.optimizer_D.zero_grad()
        loss.backward()
        self.optimizer_D.step()
        
        return loss 
        
    def step(self, L:nn.Module, reel_ab:nn.Module):
        #generate ab outputs from the generator
        fake_ab = self.generator(L)

        self.set_requires_grad(self.discriminator, True)
        d_loss = self.discriminator_step(L, reel_ab, fake_ab.detach())

        # Julien
        self.set_requires_grad(self.discriminator, False)
        g_loss, l1_loss, gan_loss = self.generator_step(L, reel_ab, fake_ab)

        return d_loss.detach(), (g_loss.detach(), l1_loss.detach(), gan_loss.detach())


def train(num_epochs, generator:nn.Module, discriminator:nn.Module, trainloader, testloader, reg_R1:bool):
    gan_train = GanTrain(generator, discriminator)

    train_g_avg_loss = []
    train_d_avg_loss = []
    test_g_avg_loss = []
    test_d_avg_loss = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_epochs):
        train_g_losses = []
        train_d_losses = []
        test_g_losses = []
        test_d_losses = []
        
        for L, ab in trainloader:
            L = L.to(device)
            ab = ab.to(device)

            g_loss, d_loss = gan_train.step(L, ab, reg_R1)

            train_g_losses.append(g_loss.detach())
            train_d_losses.append(d_loss.detach())        

        with torch.no_grad():   
            # Do not set .eval()
            for L, ab in testloader:
                L = L.to(device)
                ab = ab.to(device)
                fake_ab = generator(L)
                g_loss = gan_train.generator_loss(L, ab, fake_ab)
                d_loss = gan_train.discriminator_loss(L, ab, fake_ab)
                
                test_g_losses.append(g_loss.detach())
                test_d_losses.append(g_loss.detach())

            train_g_avg_loss.append(torch.mean(torch.Tensor(train_g_losses)))
            train_d_avg_loss.append(torch.mean(torch.Tensor(train_d_losses)))
            test_g_avg_loss.append(torch.mean(torch.Tensor(test_g_losses)))
            test_d_avg_loss.append(torch.mean(torch.Tensor(test_d_losses)))

            print('[Epoch {}/{}] '.format(i+1, num_epochs) + "\n\t--- Generator ---\n"
                  '\ttrain_loss: {:.4f} - '.format(train_g_avg_loss[-1]) +
                  'test_loss: {:.4f} - '.format(test_g_avg_loss[-1]) + "\n\t--- Discriminator ---\n"
                  'train_loss: {:.4f} - '.format(train_d_avg_loss[-1]) +
                  'test_loss: {:.4f}'.format(test_d_avg_loss[-1]))

    return train_g_avg_loss, train_d_avg_loss, test_g_avg_loss, test_d_avg_loss












# LEARNING_RATE = 0.01



# def train_G_L1(num_epochs, generator, trainloader, testloader):
#     criterion = nn.L1Loss()
#     optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)

#     train_avg_loss = []
#     test_avg_loss = []

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for i in range(num_epochs):
#         train_losses = []
#         test_losses = []
        
#         generator.train()
#         for L, ab in trainloader:
#             L = L.to(device)
#             ab = ab.to(device)

#             pred = generator(L)
#             loss = criterion(pred, ab)

#             train_losses.append(loss.detach())
#             loss.backward()

#             optimizer.step()
#             optimizer.zero_grad()
        

#         with torch.no_grad():   
#             generator.eval()
#             total = 0

#             for L, ab in testloader:
#                 L = L.to(device)
#                 ab = ab.to(device)
                
#                 pred = generator(L)
#                 loss = criterion(pred, ab)
#                 test_losses.append(loss.detach())

#                 total += len(pred)

#             print(total)

#             train_avg_loss.append(torch.mean(torch.Tensor(train_losses)))
#             test_avg_loss.append(torch.mean(torch.Tensor(test_losses)))

#             print('[Epoch {}/{}] '.format(i+1, num_epochs) +
#                   'train_loss: {:.4f} - '.format(train_avg_loss[-1]) +
#                   'test_loss: {:.4f}'.format(test_avg_loss[-1]))

#     return train_avg_loss, test_avg_loss


