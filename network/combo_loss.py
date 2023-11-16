import torch
import torch.nn as nn


class ComboLoss(nn.Module):
    """
    Combo loss function module - a combination of modified Cross Entropy (mCE) and Dice Similarity Coefficient (DSC)
    """

    def __init__(self, alpha=0.5, beta=0.5):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, masks):
        # DSC
        intersection = (outputs * masks).sum()
        union = outputs.sum() + masks.sum()
        dsc = (2. * intersection) / union if union > 0 else 1.0

        # Modified Cross Entropy (mCE)
        eps = 10e-5
        outputs = torch.clamp(outputs, eps, 1.0 - eps)
        pre_mce = self.beta * (masks * torch.log(outputs)) + (1 - self.beta) * ((1 - masks) * torch.log(1 - outputs))
        mce = -pre_mce.mean()

        return self.alpha * mce + (1 - self.alpha) * (1 - dsc)
