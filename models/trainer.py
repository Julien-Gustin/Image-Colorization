import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from PIL import Image
from torchvision import datasets, transforms, utils

LEARNING_RATE = 0.01

def train_G_L1(num_epochs, generator, trainloader, testloader):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)

    train_avg_loss = []
    test_avg_loss = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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