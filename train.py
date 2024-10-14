from __future__ import annotations
import torch
from model import *
from data_utils import ImageDatasetConfig, ImageDataset
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import json



def main(segmentation_config: SegmentationConfig):
    # segmentation_config = SegmentationConfig(
    #     n_channels=args.n_channels,
    #     n_classes=args.n_classes,
    #     alpha=args.alpha,
    #     beta=args.beta,
    #     smooth=args.smooth,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    #     batch_size=args.batch_size,
    #     epochs=args.epochs,
    #     device=args.device,
    #     seed=args.seed,
    #     betas=args.betas
    # )
    
    params = vars(segmentation_config)  

    # save the parameters dict
    with open('params.json', 'w') as fp:
        json.dump(params, fp)
    
    # set the seed
    torch.manual_seed(segmentation_config.seed)
    seed_everything(segmentation_config.seed, workers=True)

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

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:02d}",
        save_top_k=-1,  
        mode="min",
        monitor="val_loss",
        save_weights_only=True,
        every_n_epochs=1
    )
    logger = CSVLogger("logs", name="training_logs")

    trainer = Trainer(max_epochs=segmentation_config.epochs, accelerator=segmentation_config.device, devices=1, callbacks=[checkpoint_callback], deterministic=True, logger=logger)
    trainer.fit(segmentation_wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.validate(segmentation_wrapper, dataloaders=val_dataloader)
    # trainer.test(segmentation_wrapper, dataloaders=test_dataloader)



if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("n_channels", type=int, default=3, help="number of input channels")
    # parser.add_argument("n_classes", type=int, default=1, help="number of classes")
    # parser.add_argument("alpha", type=float, default=0.5, help="alpha")
    # parser.add_argument("beta", type=float, default=0.5, help="beta")
    # parser.add_argument("smooth", type=float, default=0.5, help="smooth")
    # parser.add_argument("lr", type=float, default=3e-4, help="learning rate")
    # parser.add_argument("weight_decay", type=float, default=0.01, help="weight decay")
    # parser.add_argument("batch_size", type=int, default=16, help="batch size")
    # parser.add_argument("epochs", type=int, default=15, help="number of epochs")
    # # parser.add_argument("device", type=str, default="cpu", help="device")
    # parser.add_argument("seed", type=int, default=42, help="seed")
    # # parser.add_argument("betas", type=tuple, default=(0.9, 0.999), help="betas")
    # args = parser.parse_args()

    args = SegmentationConfig(
        n_channels=3,
        n_classes=1,
        alpha=0.5,
        beta=0.5,
        smooth=0.5,
        lr=3e-4,
        weight_decay=0.01,
        batch_size=16,
        epochs=15,
        device="cpu",
        seed=42,
        betas=(0.9, 0.999)
    )
    main(args)