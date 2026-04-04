"""
模型定义 — DINOv3 backbone + 轻量化分类头
支持三种微调策略:
  1. linear_probe  — 冻结 backbone，仅训练分类头 (快速验证)
  2. lora          — LoRA 参数高效微调 (推荐，兼顾精度与显存)
  3. full_finetune — 全量微调 (极致精度，显存开销大)

DINOv3 通过 HuggingFace Transformers AutoModel 加载本地权重，
使用 pooler_output (CLS token) 作为全局特征输入分类头。
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

from config import Config


class SmileClassifier(nn.Module):
    """
    基于 DINOv3 CLS 全局特征的微笑二分类模型。
    backbone pooler_output [B, embed_dim] -> MLP 分类头 -> 2 类预测
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ── 加载 DINOv3 预训练 backbone (本地权重) ──
        print(f"[模型] 加载 DINOv3 backbone: {cfg.dinov3_model_path}")
        self.backbone = AutoModel.from_pretrained(
            cfg.dinov3_model_path,
            local_files_only=True,
        )

        # ── 分类头: CLS token -> hidden -> 2 classes ──
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.dinov3_embed_dim),
            nn.Linear(cfg.dinov3_embed_dim, cfg.classifier_hidden),
            nn.GELU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(cfg.classifier_hidden, cfg.num_classes),
        )

        # 根据微调策略配置参数冻结
        self._setup_finetune(cfg.finetune_mode)

    def _setup_finetune(self, mode: str):
        """配置微调策略"""
        if mode == "linear_probe":
            # 冻结整个 backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[模型] Linear Probe: backbone 已冻结，仅训练分类头")

        elif mode == "lora":
            # 先冻结 backbone，再注入 LoRA
            for param in self.backbone.parameters():
                param.requires_grad = False
            self._inject_lora()
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(
                f"[模型] LoRA 微调: 可训练参数 {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)"
            )

        elif mode == "full_finetune":
            # 全部参数可训练
            print("[模型] 全量微调: 所有参数均可训练")

        else:
            raise ValueError(f"未知微调模式: {mode}")

    def _inject_lora(self):
        """
        向 DINOv3 ViT 的注意力层注入 LoRA 适配器。
        使用 PEFT 库实现参数高效微调。
        """
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_target_modules,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
          x: [B, 3, 224, 224]
          return: [B, num_classes] logits
        """
        # DINOv3 AutoModel 输出: BaseModelOutputWithPooling
        # pooler_output = CLS token 的归一化特征 [B, embed_dim]
        outputs = self.backbone(pixel_values=x)
        features = outputs.pooler_output  # [B, embed_dim]

        logits = self.classifier(features)
        return logits


def build_model(cfg: Config) -> SmileClassifier:
    """构建模型并打印参数统计"""
    model = SmileClassifier(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数: {total_params:,}")
    print(f"[模型] 可训练参数: {trainable_params:,}")

    return model


def build_optimizer(model, cfg: Config):
    """
    构建优化器 — 分类头与 backbone 使用不同学习率。
    """
    # 兼容 DataParallel 包装
    raw = model.module if hasattr(model, "module") else model

    if cfg.finetune_mode == "linear_probe":
        # 仅优化分类头
        optimizer = torch.optim.AdamW(
            raw.classifier.parameters(),
            lr=cfg.lr_linear_probe,
            weight_decay=cfg.weight_decay,
        )
    else:
        # backbone (LoRA 或全量) + 分类头，差异化学习率
        backbone_params = [
            p for p in raw.backbone.parameters() if p.requires_grad
        ]
        head_params = list(raw.classifier.parameters())

        param_groups = [
            {"params": backbone_params, "lr": cfg.lr},
            {"params": head_params, "lr": cfg.lr_head},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=cfg.weight_decay
        )

    return optimizer


def load_image_processor(cfg: Config):
    """加载 DINOv3 官方 ImageProcessor (可用于推理时替代手动变换)"""
    return AutoImageProcessor.from_pretrained(
        cfg.dinov3_model_path, local_files_only=True
    )
