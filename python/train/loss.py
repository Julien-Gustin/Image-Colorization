import torch
import torch.nn as nn

from torch.nn.modules.loss import _Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class cGANLoss(_Loss): #jsp pq les loss héritent de Module : https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#L1Loss
    ## a la respondabilité de gérer les labels des true/fake prediction et de calculer la loss de prédiction labelisée
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.real_label=torch.tensor(real_label)
        self.fake_label=torch.tensor(fake_label)
        self.loss = nn.BCEWithLogitsLoss()
    


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

    Solution: https://arxiv.org/pdf/1406.2661.pdf 
    """
    def __call__(self, preds, target_is_real):
        labels = self.real_label if target_is_real else self.fake_label
        labels = labels.expand_as(preds).to(device) # on en fait un tensor de la même taille que preds full de 1 ou 0
        return self.loss(preds, labels) # attention au - ici

class R1Loss(_Loss): 
    # https://arxiv.org/pdf/1801.04406.pdf
    # https://ai.stackexchange.com/questions/25458/can-someone-explain-r1-regularization-function-in-simple-terms
    # https://github.com/ChristophReich1996/Dirac-GAN/blob/decb8283d919640057c50ff5a1ba01b93ed86332/dirac_gan/loss.py
    def __init__(self, gamma=1): 
        super().__init__()
        self.gamma = gamma

    
    def forward(self, prediction_real: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
        # gradient
        grad_real = torch.autograd.grad(outputs=prediction_real.sum(), inputs=real_sample, create_graph=True)[0]
        # regularization
        R1_loss = self.gamma * 0.5 * (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
        return R1_loss


    