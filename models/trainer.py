import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from PIL import Image
from torchvision import datasets, transforms, utils

LEARNING_RATE = 0.1

def train_G_L1(num_epochs, generator, trainloader, testloader):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)

    train_avg_loss = []
    test_avg_loss = []

    #device = 'cuda'
    device = 'cpu'

    for i in range(num_epochs):
        train_losses = []
        test_losses = []
        
        print("Train")
        for L, ab in trainloader:
            L = L.to(device)
            ab = ab.to(device)

            pred = generator(L)
            loss = criterion(pred, ab)

            print("pred.shape: ")
            print(pred.shape)
            print("pred")
            print(pred)   
            print("ab.shape: ")
            print(ab.shape)
            print("pred")
            print(ab)
            print("loss")
            print(loss)

            train_losses.append(loss.detach())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        print("Test")
        with torch.no_grad():   
            correct = 0
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