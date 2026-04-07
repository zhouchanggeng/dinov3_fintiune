"""
模型评估 — 准确率、F1、AUC、混淆矩阵、分类报告
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


def compute_metrics(
    labels: list, preds: list, probs: list
) -> dict:
    """计算核心评估指标"""
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }
    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    output_dir: str = "./outputs",
) -> dict:
    """完整模型评估 — 输出所有指标 + 可视化"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast():
            logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    # 核心指标
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 分类报告
    class_names = ["Non-Smile", "Smile"]
    report = classification_report(
        all_labels, all_preds, target_names=class_names
    )

    print("\n" + "=" * 50)
    print("Test Set Evaluation Results")
    print("=" * 50)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")

    # ── 可视化 ──
    os.makedirs(output_dir, exist_ok=True)

    # 混淆矩阵图
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC 曲线
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        all_labels, all_probs, name="Smile Classifier", ax=ax
    )
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\nVisualization saved to {output_dir}/")
    return metrics
