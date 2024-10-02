from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from model import UNet
from loss_function import BCEwithDiceLoss, dice_loss
from data_utils import ImageDatasetConfig, ImageDataset
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import Trainer
from schedulefree.adamw_schedulefree import AdamWScheduleFree


class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, bilinear: bool = False):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 64, bilinear)
        self.outc = outconv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
    

class double_conv(nn.Module):
    ''' 2 * (conv -> BN -> ReLU) '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    

class inconv(nn.Module):
    ''' double_conv '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(x)

class down(nn.Module):
    ''' maxpool -> double_conv '''
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    
    def forward(self, x):
        return self.mpconv(x)

class up(nn.Module):
    ''' upsample -> conv '''
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1) # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    

class outconv(nn.Module):
    ''' conv '''
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.conv(x)
    



@dataclass
class SegmentationConfig:
    n_channels: int
    n_classes: int
    alpha: int
    beta: int 
    smooth: float 
    lr: float 
    weight_decay: float 
    betas: tuple
    batch_size: int
    epochs: int
    device: str
    seed: int


class SegmentationWrapper(L.LightningModule):
    def __init__(self, model, config: SegmentationConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = BCEwithDiceLoss(alpha=config.alpha, beta=config.beta, smooth=config.smooth)
        self.dice_loss = dice_loss
        self.optimizer = self.configure_optimizers()

    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.train()
        optimizer.zero_grad()

        img, mask = batch
        output = self.model(img)
        loss = self.loss_fn(output, mask)

        self.log("train_loss", loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        optimizer = self.optimizers()
        optimizer.eval()

        img, mask = batch
        output = self.model(img)
        loss = self.loss_fn(output, mask)

        self.log("val_loss", loss, prog_bar=True)
    
    def configure_optimizers(self):
        return AdamWScheduleFree(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)