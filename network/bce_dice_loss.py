import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Custom loss function module - a combination of Binary Cross Entropy (BCE) and Dice Loss
    """

    def __init__(self, smooth=1):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, images, masks, smooth=1):
        # Binary Cross Entropy (BCE)
        bce_loss = self.bce(images, masks)

        # Dice
        image_soft = torch.sigmoid(images)
        image_flat, mask_flat = image_soft.view(images.size(0), -1), masks.view(masks.size(0), -1)
        intersection = (image_flat * mask_flat).sum(1)
        dice_coeff = (2. * intersection + smooth) / (image_flat.sum(1) + mask_flat.sum(1) + smooth)
        dice_loss = 1 - dice_coeff

        # Hybrid BCEDice loss
        loss = bce_loss + dice_loss.mean()

        return loss
