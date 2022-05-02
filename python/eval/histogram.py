import torch
import numpy as np

from matplotlib import pyplot as plt


class Histogrammer():
    def __init__(self, L, ab, models, batch_size, bins=25) -> None:
        self.L = L
        self.ab = ab

        ab_permuted = torch.permute(ab,(1,0,2,3))
        self.a = ab_permuted[0]*110
        print("a shape :", self.a.shape)
        self.b = ab_permuted[1]*110
        print("b shape : ", self.b.shape)

        self.models = models

        self.ab_preds = torch.empty(len(models) + 1, batch_size, 2, 256, 256) # one real ab_pred and as much ab_preds as models
        self.a_preds = torch.empty(len(models) + 1, batch_size, 256, 256)
        self.b_preds = torch.empty(len(models) + 1, batch_size, 256, 256)

        self.bins = bins
        #self.hists_a = np.empty(len(models)+1)
        #self.hists_ = np.empty(len(models)+1)
        self.hists_a = []
        self.hists_b = []
        #self.hists_a = np.zeros((len(models) + 1, ab.size()[0]))
        #self.hists_b = np.zeros((len(models) + 1, ab.size()[0]))
        self.nbr_pixels = ab.size()[0]*256*256
        self.batch_size = batch_size
        self.preds_are_done = False
        self.hists_are_done = False
    
    def compute_predictions(self):
        self.ab_preds[0] = self.ab*110
        for i, model in enumerate(self.models):
            self.ab_preds[i+1] = model(self.L).detach() # for each model, all the ab predictions of the corresponding model
            print('preds[i+1].shape',self.ab_preds[i+1].shape)

        for i, ab_pred in enumerate(self.ab_preds):
            ab_preds_permutted = torch.permute(ab_pred,(1,0,2,3))
            self.a_preds[i] = ab_preds_permutted[0]*100
            self.b_preds[i] = ab_preds_permutted[1]*100
        self.preds_are_done = True

        print("Predictions of histogrammer are done and preds shape =", self.a_preds.shape)
    

    def create_a(self):
        # return a list of models.size() of (histograms, bin_edges)
        for i, a_pred in enumerate(self.a_preds):
            print("i for hists_a ", i)
            #self.hists_a[i] = np.histogram(a_pred.reshape(-1), bins=self.bins, range=(-127, 128))
            self.hists_a.append(np.histogram(a_pred.reshape(-1), bins=self.bins, range=(-127, 128)))
    def create_b(self):
        # return a list of models.size() of (histograms, bin_edges)
        for i, b_pred in enumerate(self.b_preds):
            #self.hists_b[i] = np.histogram(b_pred.reshape(-1), bins=self.bins, range=(-127, 128))
            self.hists_b.append(np.histogram(b_pred.reshape(-1), bins=self.bins, range=(-127, 128)))

    def create_hists(self):
        if self.preds_are_done:
            self.create_a()
            self.create_b()
            self.hists_are_done = True
            print("Histograms of histogrammer are done")
        else:
            print("Predictions are not computed !")
    
    def show_a(self):
        plt.figure()
        plt.title("Histogram of a")
        plt.xlabel("a value")
        plt.ylabel("ln(pixel count/total nbr of pixels)")
        plt.xlim([-127, 128]) 
        """for hist, bin_edges in self.hists_a:
            bin_edges = bin
            plt.plot(bin_edges[0:-1], np.log(hist/self.nbr_pixels), "plop ")"""
        for hist, b in self.hists_a:
            bin_edges = b
            #plt.plot(bin_edges[0:-1], np.log(hist/self.nbr_pixels))
            plt.plot(bin_edges[0:-1], hist/self.nbr_pixels)

        plt.legend(['Ground truth', 'Dumby'])
        plt.show()


        """
        plt.figure()
        plt.title("Histogram of a")
        plt.xlabel("a value")
        plt.ylabel("ln(pixel count/total nbr of pixels)")
        plt.xlim([-127, 128]) 
        bin_edges = self.hists[0][1]
        plt.plot(bin_edges[0:-1], np.log(self.hists[0]/self.nbr_pixels))  # <- or here
        plt.show()
        """

    




