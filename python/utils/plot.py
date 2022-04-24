import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(rc={'figure.figsize':(11.7,8.27)})

class Plotter():
    def __init__(self, file_name:str, early_stop:int):

        self.d_losses_train = np.array(torch.load(file_name + "d_losses_avg_train.pt"))
        self.d_losses_val = np.array(torch.load(file_name + "d_losses_avg_val.pt"))
        self.evaluations_val = np.array(torch.load(file_name + "evaluations_avg_val.pt"))
        self.g_losses_train = np.array(torch.load(file_name + "g_losses_avg_train.pt"))
        self.g_losses_val = np.array(torch.load(file_name + "g_losses_avg_val.pt"))

        self.early_stop = early_stop

    def cGan_plot(self):
        gan_D_loss_val = self.d_losses_val[:,2]
        gan_G_loss_val = self.g_losses_val[:,2]
        gan_D_loss_train = self.d_losses_train[:,2]
        gan_G_loss_train = self.g_losses_train[:,2]
        my_plot = sns.lineplot(data=[gan_D_loss_train, gan_G_loss_train, gan_D_loss_val, gan_G_loss_val], palette=['royalblue', 'orange', 'royalblue', 'orange'])
        my_plot.lines[0].set_linestyle("-")
        my_plot.lines[1].set_linestyle("-")
        my_plot.lines[2].set_linestyle("--")
        my_plot.lines[3].set_linestyle("--")
        my_plot.set_xlabel('Epochs')
        my_plot.set_ylabel('Loss')
        my_plot.axvline(x=self.early_stop,color='red',linestyle='-.')
        plt.legend(labels=["Discriminator train", "Generator train", "Discriminator val", "Generator val"])
        plt.title('Training and Validation cGan loss')
        plt.show()

    def L1_loss_plot(self):
        L1_loss_train = self.g_losses_train[:,1]
        L1_loss_val = self.g_losses_val[:,1]
        my_plot = sns.lineplot(data=[L1_loss_train, L1_loss_val], palette=['orange',  'orange'])
        my_plot.lines[0].set_linestyle("-")
        my_plot.lines[1].set_linestyle("--")
        my_plot.axvline(x=self.early_stop,color='red',linestyle='-.')
        my_plot.set_xlabel('Epochs')
        my_plot.set_ylabel('Loss')
        plt.legend(labels=["L1 train", "L1 val"])
        plt.title('Training and Validation L1 loss Generator')
        plt.show()

    def evaluation(self):
        ssim = self.evaluations_val[:,0]
        psnr = self.evaluations_val[:,1]
        my_plot = sns.lineplot(data=[ssim], palette=['purple'])
        my_plot_2 = my_plot.twinx()
        sns.lineplot(data=[psnr], palette=['salmon'], ax=my_plot_2)
        my_plot_2.set_ylabel('PSNR')
        my_plot.legend(labels=["SSIM"], loc='upper left')
        my_plot_2.legend(labels=["PSNR"], loc='upper right')

        my_plot.lines[0].set_linestyle("-")
        my_plot.lines[1].set_linestyle("--")
        my_plot.axvline(x=self.early_stop,color='red',linestyle='-.')
        my_plot.set_xlabel('Epochs')
        my_plot.set_ylabel('SSIM')
        plt.title('SSIM and PSNR of the validation set')
        plt.show()

    def generator_loss_plot(self):
        G_loss_train = self.g_losses_train[:,0]
        G_loss_val = self.g_losses_val[:,0]
        my_plot = sns.lineplot(data=[G_loss_train, G_loss_val], palette=['orange',  'orange'])
        my_plot.lines[0].set_linestyle("-")
        my_plot.lines[1].set_linestyle("--")
        my_plot.axvline(x=self.early_stop,color='red',linestyle='-.')
        my_plot.set_xlabel('Epochs')
        my_plot.set_ylabel('$\gamma$ * L1 + cGan Loss')
        plt.legend(labels=["train", "val"])
        plt.title('Training and Validation loss of the Generator')
        plt.show()

    def discriminant_loss_plot(self, R1=False):
        if R1:
            D_loss_train = self.d_losses_train[:,0]
            D_loss_val = self.d_losses_val[:,0]

        else:
            D_loss_train = self.d_losses_train[:,2]
            D_loss_val = self.d_losses_val[:,2]
        my_plot = sns.lineplot(data=[D_loss_train, D_loss_val], palette=['royalblue',  'royalblue'])
        my_plot.lines[0].set_linestyle("-")
        my_plot.lines[1].set_linestyle("--")
        my_plot.axvline(x=self.early_stop,color='red',linestyle='-.')
        my_plot.set_xlabel('Epochs')
        my_plot.set_ylabel('Loss')
        plt.legend(labels=["train", "val"])
        plt.title('Training and Validation loss of the Discriminant')
        plt.show()