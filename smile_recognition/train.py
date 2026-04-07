"""
训练循环 — 完整的训练、验证、早停、学习率调度
"""
import os
import time
import json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

import swanlab
from config import Config
from dataset import build_dataloaders
from model import build_model, build_optimizer
from evaluate import evaluate_model, compute_metrics


class EarlyStopping:
    """早停机制 — 监控验证集指标"""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float = -float("inf")
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """学习率调度: warmup + cosine annealing"""
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.epochs * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps - warmup_steps, T_mult=1
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
) -> dict:
    """单个 epoch 训练"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """验证集评估"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


def train(cfg: Config):
    """完整训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print(f"[训练] 设备: {device}, 可用 GPU 数: {gpu_count}")

    # 数据
    loaders = build_dataloaders(cfg)

    # 模型
    model = build_model(cfg).to(device)

    # 多 GPU DataParallel
    if gpu_count > 1:
        model = nn.DataParallel(model)
        print(f"[训练] 使用 DataParallel ({gpu_count} GPUs)")

    # 优化器 & 调度器
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]))

    # 损失函数 (带 label smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # 混合精度
    scaler = GradScaler()

    # 早停
    early_stopping = EarlyStopping(cfg.patience, cfg.min_delta)

    # 训练日志
    history = {"train": [], "val": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(cfg.output_dir, "best_model.pth")

    # SwanLab
    swanlab.init(
        project="smile-recognition",
        experiment_name=f"{cfg.finetune_mode}-{time.strftime('%m%d_%H%M')}",
        config=vars(cfg),
        logdir=os.path.join(cfg.output_dir, "swanlog"),
        mode="local",
    )

    print(f"\n[训练] 开始训练 — {cfg.finetune_mode} 模式, {cfg.epochs} epochs")
    print("=" * 60)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # 训练
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scheduler, scaler, device
        )

        # 验证
        val_metrics = validate(model, loaders["val"], criterion, device)

        elapsed = time.time() - t0
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # SwanLab 记录
        swanlab.log({
            "train/loss": train_metrics["loss"],
            "train/accuracy": train_metrics["accuracy"],
            "val/loss": val_metrics["loss"],
            "val/accuracy": val_metrics["accuracy"],
            "val/f1": val_metrics["f1"],
            "val/auc": val_metrics["auc"],
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        # 保存最优模型
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(model, optimizer, epoch, val_metrics, best_model_path, cfg)
            print(f"  ✓ 保存最优模型 (val_acc={best_val_acc:.4f})")

        # 早停检查
        if early_stopping.step(val_metrics["accuracy"]):
            print(f"\n[早停] 验证集指标连续 {cfg.patience} 个 epoch 未提升，停止训练")
            break

    # ── 测试集最终评估 ──
    print("\n" + "=" * 60)
    print("[评估] 加载最优模型，在测试集上评估")
    raw_model = model.module if hasattr(model, "module") else model
    load_checkpoint(raw_model, best_model_path)
    test_metrics = evaluate_model(model, loaders["test"], device)

    # 保存训练历史
    history_path = os.path.join(cfg.output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    swanlab.finish()
    return model, test_metrics


def save_checkpoint(model, optimizer, epoch, metrics, path, cfg):
    """保存检查点 (兼容 DataParallel)"""
    # 如果是 DataParallel 包装，取 .module
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "dinov3_model_path": cfg.dinov3_model_path,
            "finetune_mode": cfg.finetune_mode,
            "lora_r": cfg.lora_r,
        },
    }
    torch.save(state, path)


def load_checkpoint(model, path):
    """加载检查点"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"  已加载检查点: epoch={checkpoint['epoch']}")
    return checkpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="微笑识别模型训练")
    parser.add_argument(
        "--mode",
        type=str,
        default="lora",
        choices=["linear_probe", "lora", "full_finetune"],
        help="微调策略",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.finetune_mode = args.mode
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr
    cfg.__post_init__()

    train(cfg)
