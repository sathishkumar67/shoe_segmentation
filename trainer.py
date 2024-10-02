
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from loss_function import BCEwithDiceLoss, dice_loss
from data_utils import ImageDatasetConfig, ImageDataset
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer

def main(segmentation_config: SegmentationConfig):
    # set the seed
    torch.manual_seed(segmentation_config.seed)

    train_dataset_config = ImageDatasetConfig()
    train_dataset = ImageDataset(image_ds_config=train_dataset_config)
    train_dataloader = DataLoader(train_dataset, batch_size=segmentation_config.batch_size, shuffle=True)

    val_dataset_config = ImageDatasetConfig()
    val_dataset = ImageDataset(image_ds_config=val_dataset_config)
    val_dataset_config.mode, val_dataset_config.augment = "val", False
    val_dataloader = DataLoader(val_dataset, batch_size=segmentation_config.batch_size, shuffle=False)

    # test_dataset_config = ImageDatasetConfig()
    # test_dataset = ImageDataset(image_ds_config=test_dataset_config)
    # test_dataset_config.mode, test_dataset_config.augment = "test", False
    # test_dataloader = DataLoader(test_dataset, batch_size=segmentation_config.batch_size, shuffle=False)

    model = UNet(n_channels=segmentation_config.n_channels, n_classes=segmentation_config.n_classes)
    segmentation_wrapper = SegmentationWrapper(model, segmentation_config)
    trainer = Trainer(max_epochs=segmentation_config.epochs, accelerator=segmentation_config.device)
    trainer.fit(segmentation_wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    seg_config = SegmentationConfig(
        n_channels=3,
        n_classes=1,
        alpha=0.5,
        beta=0.5,
        smooth=0.5,
        lr=0.001,
        weight_decay=0.0001,
        betas=(0.9, 0.999),
        batch_size=8,
        epochs=1,
        device="cpu",
        seed=42
    )

    main(seg_config)