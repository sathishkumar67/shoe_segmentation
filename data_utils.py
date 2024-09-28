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
    def __init__(self, image_ds_config) -> None:
        super().__init__()

        self.foreground_directory = image_ds_config.foreground_dir
        self.background_directory = image_ds_config.background_dir
        self.mode = image_ds_config.mode
        self.image_size = image_ds_config.image_size
        self.augment = image_ds_config.augment
        self.augment_prob = image_ds_config.augment_prob
        self.rotation_degree = [0, 90, 180, 270]

        self.train_images = list(map(lambda x: f"{self.foreground_directory}train/{x}", os.listdir(f"{self.foreground_directory}/train")))
        self.val_images = list(map(lambda x: f"{self.foreground_directory}val/{x}", os.listdir(f"{self.foreground_directory}/val")))
        self.test_images = list(map(lambda x: f"{self.foreground_directory}test/{x}", os.listdir(f"{self.foreground_directory}/test")))
        self.train_background_images = list(map(lambda x: f"{self.background_directory}train/{x}", os.listdir(f"{self.background_directory}/train")))
        self.val_background_images = list(map(lambda x: f"{self.background_directory}val/{x}", os.listdir(f"{self.background_directory}/val")))

    def __getitem__(self, index):
        if self.mode == "train":
            img_path = self.train_images[index]
        elif self.mode == "val":
            img_path = self.val_images[index]
        else:
            img_path = self.test_images[index]

        return self.transform_image(img_path, self.augment)
    
    def __len__(self):
        if self.mode == "train":
            return len(self.train_images)
        elif self.mode == "val":
            return len(self.val_images)
        else:
            return len(self.test_images)
    
    def transform_image(self, img_path: str, augment: bool):
        image_alpha = Image.open(img_path)
        assert str(image_alpha.mode) == 'RGBA'
        x, y = image_alpha.size
        aspect_ratio = y / x
        ch_r, ch_g, ch_b, ch_a = image_alpha.split()
        img = Image.merge('RGB', (ch_r, ch_g, ch_b))
        mask = ch_a
        
        if self.mode == "train":
            bg = Image.open(self.train_background_images[random.randint(0, len(self.train_background_images)-1)])
            bg = bg.resize(img.size)
            bg.paste(img, mask=mask)
        else:
            bg = Image.open(self.val_background_images[random.randint(0, len(self.val_background_images)-1)])
            bg = bg.resize(img.size)
            bg.paste(img, mask=mask)

        img = bg
            
        if augment and random.random() < self.augment_prob:
            transform = list()
            resize_range = random.randint(300, 320)
            transform.append(T.Resize((int(resize_range * aspect_ratio), resize_range)))
            rot_deg = self.rotation_degree[random.randint(0, 3)]
            if rot_deg == 90 or rot_deg == 270:
                aspect_ratio = 1 / aspect_ratio
            transform.append(T.RandomRotation((rot_deg, rot_deg)))
            rot_range = random.randint(-10, 10)
            transform.append(T.RandomRotation((rot_range, rot_range)))
            crop_range = random.randint(270, 300)
            transform.append(T.CenterCrop((int(crop_range * aspect_ratio), crop_range)))
            transform = T.Compose(transform)

            img = transform(img)
            mask = transform(mask)

            transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)

            img = transform(img)

            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            
        transform = list()
        transform.append(T.Resize((self.image_size, self.image_size)))
        transform.append(T.ToTensor())
        transform = T.Compose(transform)

        img = transform(img)
        mask = transform(mask)

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
    