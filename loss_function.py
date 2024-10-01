from __future__ import annotations
import torch.nn as nn

class BCELoss2d(nn.Module):
    
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)
        return self.bce_loss(predict, target)

def dice_coeff(predict, target):
    smooth = 0.001
    batch_size = predict.size(0)
    predict = (predict > 0.5).float()
    m1 = predict.view(batch_size, -1)
    m2 = target.view(batch_size, -1)
    intersection = (m1 * m2).sum(-1)
    return ((2.0 * intersection + smooth) / (m1.sum(-1) + m2.sum(-1) + smooth)).mean()


# dice loss = 1 - dice_coeff
def dice_loss(predict, target):
    smooth = 0.001
    batch_size = predict.size(0)
    predict = (predict > 0.5).float()
    m1 = predict.view(batch_size, -1)
    m2 = target.view(batch_size, -1)
    intersection = (m1 * m2).sum(-1)
    return 1 - ((2.0 * intersection + smooth) / (m1.sum(-1) + m2.sum(-1) + smooth)).mean()


class BCEwithDiceLoss(nn.Module):
    
    def __init__(self, alpha: int = 0.5, beta: int = 0.5, smooth: float = 1e-5):
        super(BCEwithDiceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, predict, target):
        # calculate bce loss
        bce_loss = self.bce_loss(predict.view(-1), target.view(-1))
        
        # calculate dice loss
        batch_size = predict.size(0)
        predict = (predict > 0.5).float()
        m1 = predict.view(batch_size, -1)
        m2 = target.view(batch_size, -1)
        intersection = (m1 * m2).sum(-1)
        dice_loss =  1 - ((2.0 * intersection + self.smooth) / (m1.sum(-1) + m2.sum(-1) + self.smooth)).mean()
        
        # combine bce and dice loss
        return self.alpha * bce_loss + self.beta * dice_loss