"""
数据集与数据增强 — 严格匹配 DINOv3 预训练规范
包含适配表情识别任务的轻量化增强策略
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config


class SmileDataset(Dataset):
    """
    微笑/非微笑二分类数据集。
    支持两种加载方式:
      1. 从 split 文件 (train.txt / val.txt / test.txt) 加载
      2. 从目录结构 (0/ 1/) 加载
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        transform=None,
    ):
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        split_file = os.path.join(cfg.aligned_dir, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        self.samples.append((parts[0], int(parts[1])))
        else:
            # fallback: 从目录加载
            for label in [0, 1]:
                label_dir = os.path.join(cfg.aligned_dir, str(label))
                if not os.path.isdir(label_dir):
                    continue
                for fname in sorted(os.listdir(label_dir)):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append(
                            (os.path.join(label_dir, fname), label)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            # 返回黑图作为 fallback
            image = np.zeros(
                (self.cfg.image_size, self.cfg.image_size, 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = cv2.resize(image, (self.cfg.image_size, self.cfg.image_size))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)


def get_train_transform(cfg: Config) -> A.Compose:
    """
    训练集增强 — 适配表情识别的轻量化策略:
    - 水平翻转 (微笑左右对称)
    - 轻微旋转/仿射 (模拟头部姿态变化)
    - 亮度/对比度微调 (模拟光照变化)
    - 严格匹配 DINOv3 ImageNet 归一化
    """
    return A.Compose([
        A.Resize(cfg.image_size, cfg.image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3
        ),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(
            max_holes=1, max_height=32, max_width=32,
            min_holes=1, min_height=16, min_width=16, p=0.1
        ),
        A.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
        ToTensorV2(),
    ])


def get_val_transform(cfg: Config) -> A.Compose:
    """验证/测试集变换 — 仅 resize + DINOv3 归一化"""
    return A.Compose([
        A.Resize(cfg.image_size, cfg.image_size),
        A.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
        ToTensorV2(),
    ])


def build_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """构建 train/val/test DataLoader"""
    train_ds = SmileDataset(cfg, "train", get_train_transform(cfg))
    val_ds = SmileDataset(cfg, "val", get_val_transform(cfg))
    test_ds = SmileDataset(cfg, "test", get_val_transform(cfg))

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
    }

    print(f"数据集大小 — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return loaders
