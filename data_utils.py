from __future__ import annotations
import os
import random
import numpy as np
from PIL import Image
import pydensecrf
from dataclasses import dataclass
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass
class ImageDatasetConfig:
    foreground_dir: str = "shoe_dataset/"
    background_dir: str = "shoe_dataset/bg/"
    mode: str = "train"
    image_size: int = 256
    augment: bool = True
    augment_prob: float = 0.5

img_ds_config = ImageDatasetConfig()


class ImageDataset(Dataset):
    def __init__(self, image_ds_config: ImageDatasetConfig) -> None:
        super().__init__()

        self.foreground_directory = image_ds_config.foreground_dir
        self.background_directory = image_ds_config.background_dir
        self.mode = image_ds_config.mode
        self.image_size = image_ds_config.image_size
        self.augment = image_ds_config.augment
        self.augment_prob = image_ds_config.augment_prob
        self.rotation_degree = [0, 90, 180, 270]

        # Load file paths lazily when needed
        self.image_paths = self._load_image_paths()

        # Pre-load background images to avoid repeated I/O operations
        self.background_images = self._load_background_images()

        # Prepare common transforms
        self.base_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])


    def __len__(self):
        return len(self.image_paths)
        

    def _load_image_paths(self):
        if self.mode == "train":
            return [os.path.join(self.foreground_directory, "train", x) for x in os.listdir(os.path.join(self.foreground_directory, "train"))]
        elif self.mode == "val":
            return [os.path.join(self.foreground_directory, "val", x) for x in os.listdir(os.path.join(self.foreground_directory, "val"))]
        else:
            return [os.path.join(self.foreground_directory, "test", x) for x in os.listdir(os.path.join(self.foreground_directory, "test"))]


    def _load_background_images(self):
        bg_dir = os.path.join(self.background_directory, self.mode)
        return [os.path.join(bg_dir, x) for x in os.listdir(bg_dir)]


    def __getitem__(self, index):
        img_path = self.image_paths[index]

        # Load and transform image and mask
        img, mask = self._load_and_process_image(img_path)

        # Apply augmentations if enabled
        if self.mode == "train" and self.augment and random.random() < self.augment_prob:
            img, mask = self._apply_augmentation(img, mask)

        # Apply final resizing and convert to tensor
        img = self.base_transform(img)
        mask = self.base_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.image_paths)

    def _load_and_process_image(self, img_path: str):
        # Load the foreground image with alpha channel
        image_alpha = Image.open(img_path)
        assert image_alpha.mode == 'RGBA', "Image should be RGBA"

        img = Image.merge('RGB', image_alpha.split()[:3])
        mask = image_alpha.split()[-1]  # Alpha channel as mask

        # Select and resize a random background image
        bg_img = Image.open(random.choice(self.background_images)).resize(img.size)

        # Composite the foreground over the background using the mask
        bg_img.paste(img, mask=mask)

        return bg_img, mask

    def _apply_augmentation(self, img, mask):
        """Apply the same augmentations to both the image and the mask."""
        
        # Random Rotation
        angle = random.uniform(-10, 10)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)

        # Random Color Jitter (only applied to img, not mask, since mask is not RGB)
        color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)
        img = color_jitter(img)

        # Random Horizontal Flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Random Vertical Flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        return img, mask




def dense_crf(inputs, predict_probs):
    h = predict_probs.shape[0]
    w = predict_probs.shape[1]
    
    predict_probs = np.expand_dims(predict_probs, 0)
    predict_probs = np.append(1 - predict_probs, predict_probs, axis=0)
    
    d = pydensecrf.densecrf.DenseCRF2D(w, h, 2)
    U = -np.log(predict_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    inputs = np.ascontiguousarray(inputs)
    
    d.setUnaryEnergy(U)
    
    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=inputs, compat=10)
    
    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    
    return Q




# class TestImageDataset(Dataset):
    
#     def __init__(self, fdir, imsize):
#         self._fdir = fdir
#         self._imsize = imsize
#         self._impaths = list(map(lambda x: os.path.join(fdir, x), os.listdir(fdir)))
#         Transform = list()
#         Transform.append(T.Resize((self._imsize, self._imsize)))
#         Transform.append(T.ToTensor())
#         Transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#         Transform = T.Compose(Transform)
#         self._dataset = list()
#         self._osize = list()
#         for file in self._impaths:
#             img = Image.open(file)
#             assert str(img.mode) == 'RGB'
#             self._osize.append(img.size)
#             img = Transform(img)
#             self._dataset.append(img)
#         print("image count in test path: {}".format(len(self._impaths)))
    
#     def __getitem__(self, index):
#         return index, self._dataset[index]
    
#     def __len__(self):
#         return len(self._impaths)
    
#     def save_img(self, index, predict, use_crf):
#         predict = predict.squeeze().cpu().numpy()
#         if use_crf:
#             inputs = self._dataset[index].permute(1, 2, 0).numpy()
#             predict = dense_crf(np.array(inputs).astype(np.uint8), predict)
#         predict = np.array((predict > 0.5) * 255).astype(np.uint8)
#         mask = Image.fromarray(predict, mode='L')
#         mask = mask.resize(self._osize[index])
#         fg = Image.new('RGB', self._osize[index], (0, 0, 0))
#         bg = Image.new('RGB', self._osize[index], (255, 255, 255))
#         bg.paste(fg, mask=mask)
#         bg.save('./predicts/{:s}'.format(os.path.split(self._impaths[index])[-1]))
    