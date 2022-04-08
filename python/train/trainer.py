from matplotlib.pyplot import axis
import torch
import torch.nn as nn

from .loss import cGANLoss

# https://arxiv.org/pdf/1406.2661.pdf (train GAN)
class GanTrain():   
    def __init__(self, generator:nn.Module, discriminator:nn.Module) -> None:
        self.generator = generator
        self.discriminator = discriminator

        self.cgan_loss = cGANLoss()
        self.l1_loss = nn.L1Loss()

        # TODO: hyperparameters in arguments
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.gamma = 0.5

    def set_requires_grad(self, model:nn.Module, requires_grad:bool):
        for param in model.parameters():
            param.requires_grad = requires_grad 

    def generator_loss(self, L:torch.Tensor, ab_reel:torch.Tensor):
        """ Compute the loss of the generator according to a weighted sum of L1 loss and GAN loss """
        ab_fake = self.generator(L)
        Lab_fake = torch.cat((L, ab_fake), axis=1)
        discriminator_confidence = self.discriminator(Lab_fake)

        l1_loss = self.l1_loss(ab_fake, ab_reel)
        gan_loss = self.cgan_loss(discriminator_confidence, True) # trick

        loss = l1_loss * self.gamma + gan_loss

        return loss

    def generator_step(self, L:torch.Tensor, ab_reel:torch.Tensor):
        """ Perform a one step generator training """
        loss = self.generator_loss(L, ab_reel)
        loss.backwards()

        self.optimizer_G.step()
        self.optimizer_G.zero_grad()

        return loss


    def discriminator_backward(self):
        # Houyon TODO
        pass

    def step(self, L:nn.Module, ab_reel:nn.Module):
        # Houyon TODO

        d_loss = self.discriminator_backward()

        # Julien
        self.set_requires_grad(self.discriminator, False)
        g_loss = self.generator_step(L, ab_reel)
        self.set_requires_grad(self.discriminator, True)

        return d_loss.detach(), g_loss.detach()


def train(num_epochs, generator:nn.Module, discriminator:nn.Module, trainloader, testloader):
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

            g_loss, d_loss = gan_train.step(L, ab)

            train_g_losses.append(g_loss.detach())
            train_d_losses.append(d_loss.detach())        

        with torch.no_grad():   
            # Do not set .eval()
            for L, ab in testloader:
                L = L.to(device)
                ab = ab.to(device)

                g_loss = gan_train.generator_loss(L, ab)
                d_loss = gan_train.discriminator_loss(L, ab)
                
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












LEARNING_RATE = 0.01



def train_G_L1(num_epochs, generator, trainloader, testloader):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)

    train_avg_loss = []
    test_avg_loss = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_epochs):
        train_losses = []
        test_losses = []
        
        generator.train()
        for L, ab in trainloader:
            L = L.to(device)
            ab = ab.to(device)

            pred = generator(L)
            loss = criterion(pred, ab)

            train_losses.append(loss.detach())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        

        with torch.no_grad():   
            generator.eval()
            total = 0

            for L, ab in testloader:
                L = L.to(device)
                ab = ab.to(device)
                
                pred = generator(L)
                loss = criterion(pred, ab)
                test_losses.append(loss.detach())

                total += len(pred)

            print(total)

            train_avg_loss.append(torch.mean(torch.Tensor(train_losses)))
            test_avg_loss.append(torch.mean(torch.Tensor(test_losses)))

            print('[Epoch {}/{}] '.format(i+1, num_epochs) +
                  'train_loss: {:.4f} - '.format(train_avg_loss[-1]) +
                  'test_loss: {:.4f}'.format(test_avg_loss[-1]))

    return train_avg_loss, test_avg_loss


