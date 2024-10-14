
from __future__ import annotations
import torch
from model import *
from data_utils import ImageDatasetConfig, ImageDataset
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
# from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import json



def main(segmentation_config: SegmentationConfig):

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
    model.apply(init_weights)
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
    seg_config = SegmentationConfig(
                n_channels=3,
                n_classes=1,
                alpha=0.5,
                beta=0.5,
                smooth=0.5,
                lr=3e-4,
                weight_decay=0.01,
                batch_size=16,
                epochs=15,
                device="cuda",
                seed=42,
                betas=(0.9, 0.999)
            )
    
    
    # set the seed
    torch.manual_seed(seg_config.seed)

    # train the model with the hyperparameters
    train_loss, val_loss = main(seg_config)

    # save the training and validation loss
    np.save("training_loss.npy", np.array(train_loss))
    np.save("validation_loss.npy", np.array(val_loss))

    # save the parameters dict
    # with open('params.json', 'w') as fp:
    #     json.dump(params, fp)