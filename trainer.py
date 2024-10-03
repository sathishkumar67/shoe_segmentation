
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from loss_function import BCEwithDiceLoss, dice_loss
from data_utils import ImageDatasetConfig, ImageDataset
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np


def main(segmentation_config: SegmentationConfig):
    # set the seed
    torch.manual_seed(segmentation_config.seed)

    train_dataset_config = ImageDatasetConfig()
    train_dataset = ImageDataset(image_ds_config=train_dataset_config)
    train_dataloader = DataLoader(train_dataset, batch_size=segmentation_config.batch_size, shuffle=True)

    val_dataset_config = ImageDatasetConfig()
    val_dataset_config.mode, val_dataset_config.augment = "val", False
    val_dataset = ImageDataset(image_ds_config=val_dataset_config)
    val_dataloader = DataLoader(val_dataset, batch_size=segmentation_config.batch_size, shuffle=False)

    # test_dataset_config = ImageDatasetConfig()
    # test_dataset = ImageDataset(image_ds_config=test_dataset_config)
    # test_dataset_config.mode, test_dataset_config.augment = "test", False
    # test_dataloader = DataLoader(test_dataset, batch_size=segmentation_config.batch_size, shuffle=False)

    model = UNet(n_channels=segmentation_config.n_channels, n_classes=segmentation_config.n_classes)
    segmentation_wrapper = SegmentationWrapper(model, segmentation_config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:02d}",
        save_top_k=-1,  
        mode="min",
        monitor="val_loss",
        save_weights_only=True,
        every_n_epochs=1
    )

    trainer = Trainer(max_epochs=segmentation_config.epochs, accelerator=segmentation_config.device, callbacks=[checkpoint_callback])
    trainer.fit(segmentation_wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    np.save("train_loss.npy", np.array(segmentation_wrapper.train_loss))
    np.save("val_loss.npy", np.array(segmentation_wrapper.val_loss))




if __name__ == "__main__":

    # # create a grid search function to find the best hyperparameters
    # def grid_search(segmentation_config: SegmentationConfig):
    #     hyperparameters = {
    #         "alpha": [0.2, 0.5, 0.8],
    #         "beta": [0.2, 0.5, 0.8],
    #         "smooth": [0.1, 0.5, 1.0],
    #         "lr": [1e-4, 3e-4, 1e-3],
    #         "weight_decay": [0.0001, 0.001, 0.01],
    #         "betas": [(0.8, 0.99), (0.9, 0.999), (0.95, 0.9995)],
    #         "batch_size": [4, 8, 16],
    #     }
    #     for hyperparameter_name, hyperparameter_values in hyperparameters.items():
    #         for hyperparameter_value in hyperparameter_values:
    #             new_seg_config = dataclasses.replace(segmentation_config, **{hyperparameter_name: hyperparameter_value})
    #             main(new_seg_config)



    seg_config = SegmentationConfig(
        n_channels=3,
        n_classes=1,
        alpha=0.5,
        beta=0.5,
        smooth=0.5,
        lr=3e-4,
        weight_decay=0.0001,
        betas=(0.9, 0.999),
        batch_size=8,
        epochs=5,
        device="cuda",
        seed=42
    )

    main(seg_config)