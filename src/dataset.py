"""
PyTorch Dataset for Turf Segmentation
======================================
Loads image + mask patches, applies augmentations for training.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class TurfDataset(Dataset):
    """
    Dataset that loads 512x512 image patches and their binary masks.

    Directory structure expected:
        root/
          images/  *.png  (RGB)
          masks/   *.png  (grayscale, 0=bg, 255=turf)
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 512):
        self.split      = split
        self.image_size = image_size
        self.is_train   = (split == "train")

        img_dir  = Path(root) / split / "images"
        mask_dir = Path(root) / split / "masks"

        self.img_paths  = sorted(img_dir.glob("*.png"))
        self.mask_paths = sorted(mask_dir.glob("*.png"))

        assert len(self.img_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.img_paths)} images vs {len(self.mask_paths)} masks"
        assert len(self.img_paths) > 0, f"No images found in {img_dir}"

        print(f"[Dataset/{split}] {len(self.img_paths)} samples loaded")

        # ImageNet normalization (SegFormer pretrained on ImageNet)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.img_paths)

    def _augment(self, image: Image.Image, mask: Image.Image):
        """Apply consistent random augmentations to image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # Random rotation (90-degree increments only, no interpolation artifacts)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask, angle)

        # Color jitter (image only, not mask)
        jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        image = jitter(image)

        return image, mask

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")  # grayscale

        # Resize to ensure consistent size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask  = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # Augment training data
        if self.is_train:
            image, mask = self._augment(image, mask)

        # Convert to tensors
        image = TF.to_tensor(image)             # (3, H, W) float32 [0,1]
        image = self.normalize(image)

        mask_arr = np.array(mask)               # (H, W) uint8 {0, 255}
        mask_tensor = torch.from_numpy((mask_arr > 127).astype(np.int64))  # {0, 1}

        return image, mask_tensor


def get_dataloaders(patches_dir: str, batch_size: int = 4, num_workers: int = 0):
    """Build train and val DataLoaders."""
    train_ds = TurfDataset(patches_dir, split="train")
    val_ds   = TurfDataset(patches_dir, split="val")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader
