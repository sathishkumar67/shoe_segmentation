
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from data_utils import ImageDatasetConfig, ImageDataset
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
# from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import ParameterSampler


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
    # test_dataset_config.mode, test_dataset_config.augment = "test", False
    # test_dataset = ImageDataset(image_ds_config=test_dataset_config)
    # test_dataloader = DataLoader(test_dataset, batch_size=segmentation_config.batch_size, shuffle=False)

    model = UNet(n_channels=segmentation_config.n_channels, n_classes=segmentation_config.n_classes)
    segmentation_wrapper = SegmentationWrapper(model, segmentation_config)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="checkpoints",
    #     filename="{epoch:02d}",
    #     save_top_k=-1,  
    #     mode="min",
    #     monitor="val_loss",
    #     save_weights_only=True,
    #     every_n_epochs=1
    # )

    trainer = Trainer(max_epochs=segmentation_config.epochs, accelerator=segmentation_config.device, devices=1)
    trainer.fit(segmentation_wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    train_loss = segmentation_wrapper.train_loss
    val_loss = segmentation_wrapper.val_loss

    return train_loss, val_loss



if __name__ == "__main__":

    # Your hyperparameters
    hyperparameters = {
        "alpha_beta": [(0.4, 0.6), (0.5, 0.5), (0.6, 0.4)],  # alpha and beta for the loss function
        "lr": [1e-3, 1e-4, 1e-5],  # learning rate
        "weight_decay": [1e-3, 1e-4, 1e-5],  # weight decay
        "epochs": [5, 10, 15]  # epochs
    }

    # Generate 20 random hyperparameter combinations
    random_combinations = list(ParameterSampler(hyperparameters, n_iter=5, random_state=1337))
    print(random_combinations)
    # Random search for find best hyperparameters
    def random_search(hyperparameters):
        training_loss = []
        validation_loss = []
        combinations = []
        for params in hyperparameters:
            seg_config = SegmentationConfig(
                n_channels=3,
                n_classes=1,
                alpha=params["alpha_beta"][0],
                beta=params["alpha_beta"][1],
                smooth=0.5,
                lr=params["lr"].item(),
                weight_decay=params["weight_decay"].item(),
                betas=params["betas"],
                batch_size=16,
                epochs=params["epochs"],
                device="cuda",
                seed=42
            )

            torch.manual_seed(seg_config.seed)
            train_loss, val_loss = main(seg_config)

            training_loss.append(train_loss)
            validation_loss.append(val_loss)
            combinations.append(params)

        np.save("training_loss.npy", training_loss)
        np.save("validation_loss.npy", validation_loss)
        np.save("combinations.npy", combinations)

    random_search(random_combinations)