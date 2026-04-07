"""
全局配置 — 微笑/非微笑二分类表情识别
基于 DINOv3 自监督视觉大模型 + InsightFace 人脸检测对齐
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── 路径 ──
    data_root: str = "./data/kaggle-genki4k"      # GENKI-4K 数据集根目录 (smile/ non_smile/)
    rafdb_root: str = "./data/RAF-DB"             # RAF-DB 数据集根目录
    aligned_dir: str = "./data/GENKI-4K_aligned"  # 对齐后人脸存储目录
    output_dir: str = "./outputs"                  # 模型/日志输出目录

    # ── DINOv3 模型 ──
    # 本地预训练权重目录 (HuggingFace 格式)
    dinov3_model_path: str = "./dinov3-vitl16-pretrain-lvd1689m"
    dinov3_embed_dim: int = 1024   # ViT-L hidden_size
    image_size: int = 224
    patch_size: int = 16
    num_register_tokens: int = 4

    # ── 微调策略 ──
    # "linear_probe" | "lora" | "full_finetune"
    finetune_mode: str = "lora"

    # LoRA 参数
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj"]  # DINOv3 ViT 注意力投影层
    )

    # ── 分类头 ──
    num_classes: int = 2
    classifier_hidden: int = 256
    classifier_dropout: float = 0.3

    # ── 训练 ──
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4               # LoRA / full finetune
    lr_head: float = 1e-3          # 分类头学习率
    lr_linear_probe: float = 1e-3  # linear probe 学习率
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    label_smoothing: float = 0.1

    # 早停
    patience: int = 7
    min_delta: float = 1e-4

    # ── 数据 ──
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 4
    seed: int = 42

    # ── InsightFace ──
    face_det_size: int = 640       # 检测输入尺寸
    face_align_size: int = 112     # 对齐输出尺寸 (最终会 resize 到 image_size)

    # ── DINOv3 归一化 (ImageNet 预训练规范) ──
    norm_mean: tuple = (0.485, 0.456, 0.406)
    norm_std: tuple = (0.229, 0.224, 0.225)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.aligned_dir, exist_ok=True)
