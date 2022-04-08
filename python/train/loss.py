import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
class cGANValue(_Loss): #jsp pq les loss héritent de Module : https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#L1Loss
    ## a la respondabilité de gérer les labels des true/fake prediction et de calculer la loss de prédiction labelisée
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.real_label=torch.tensor(real_label)
        self.fake_label=torch.tensor(fake_label)
        self.loss = nn.BCELoss() # nn.BCEWithLogitsLoss() si on retire la sigmoide du discriminator
    


    """
    (Course)
    For generator: 
        loss = cGANValue(...)

    For discrimantor:
        loss = -cGANValue(...)

    Vs

    (Source code)
    For generator: 
        loss = cGANValue(fake, True)

    For discrimantor:
        loss = cGANValue(fake, False)
    """
    def __call__(self, preds, target_is_real):
        labels = self.real_label if target_is_real else self.fake_label
        labels = labels.expand_as(preds) # on en fait un tensor de la même taille que preds full de 1 ou 0
        return -self.loss(preds, labels) # attention au - ici

