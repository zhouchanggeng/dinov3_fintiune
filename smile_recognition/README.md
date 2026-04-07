# 微笑识别 — DINOv3 + InsightFace

基于 Meta DINOv3 (arxiv: 2508.10104) 自监督视觉大模型的微笑/非微笑二分类表情识别系统。

## 核心流程

```
原图 → InsightFace 人脸检测对齐 → DINOv3 特征提取 → 分类头 → 微笑/非微笑
```

## 项目结构

```
smile_recognition/
├── config.py              # 全局配置
├── prepare_data.py        # 人脸检测、对齐与数据集划分
├── dataset.py             # Dataset & 数据增强
├── model.py               # DINOv3 backbone + 分类头 (Linear Probe / LoRA / Full)
├── train.py               # 训练循环 (含早停、学习率调度、多GPU支持)
├── evaluate.py            # 评估指标 & 可视化
├── inference.py           # 单张图片端到端推理
├── requirements.txt       # 依赖
└── dinov3-vitl16-pretrain-lvd1689m/  # DINOv3 本地预训练权重
```

## 模型

使用本地预训练权重 `dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, 300M 参数, embed_dim=1024, patch_size=16, 4 register tokens, RoPE)，
通过 HuggingFace Transformers `AutoModel` 加载，提取 `pooler_output` (CLS token) 作为全局特征。

## 环境安装

```bash
conda create -n smile python=3.11 -y
conda activate smile
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> DINOv3 需要 `transformers>=4.56.0`

## 数据准备

数据集为 kaggle-genki4k 格式 (smile / non_smile 目录)，解压 `GENKI-4K.zip` 后执行人脸检测对齐与数据集划分:

```bash
unzip GENKI-4K.zip -d data/
python prepare_data.py
```

处理流程:
1. InsightFace (buffalo_l) 人脸检测 + 5 点关键点仿射对齐 (ArcFace 标准模板)
2. 对齐后人脸按 label 存储至 `data/GENKI-4K_aligned/{0,1}/`
3. 分层划分 train/val/test (80%/10%/10%)

## 训练

三种微调策略可选:

```bash
# 推荐: LoRA 参数高效微调 (兼顾精度与显存)
CUDA_VISIBLE_DEVICES=0,2,4,6 python train.py --mode lora

# 快速验证: 线性探测 (仅训练分类头)
CUDA_VISIBLE_DEVICES=0 python train.py --mode linear_probe

# 极致精度: 全量微调 (需要较大显存)
CUDA_VISIBLE_DEVICES=0,2,4,6 python train.py --mode full_finetune --lr 5e-5
```

训练特性:
- Warmup + Cosine Annealing 学习率调度
- 混合精度训练 (AMP)
- 梯度裁剪 (max_norm=1.0)
- 早停机制 (patience=7)
- Label Smoothing (0.1)
- 多 GPU DataParallel 支持

## 推理

```bash
# 单张图片推理
python inference.py path/to/image.jpg --checkpoint ./outputs/best_model.pth

# 带可视化输出
python inference.py path/to/image.jpg --save result.jpg
```

## 评估指标

训练完成后自动在测试集上输出:
- Accuracy / F1 / Precision / Recall / AUC
- Confusion Matrix (保存为 `outputs/confusion_matrix.png`)
- ROC Curve (保存为 `outputs/roc_curve.png`)

## 微调策略对比

| 策略 | 可训练参数 | 显存占用 | 精度 | 适用场景 |
|------|-----------|---------|------|---------|
| Linear Probe | ~0.3M | 低 | ★★★ | 快速验证 |
| LoRA (r=8, target: q_proj/v_proj) | ~2M | 中 | ★★★★ | **推荐** |
| Full Finetune | ~300M | 高 | ★★★★★ | 极致精度 |

## 进阶优化方向

- 人脸精准仿射对齐 (3D 姿态估计 + 正面化)
- 人脸专用数据增强 (遮挡模拟、表情混合)
- 大规模人脸数据集域适配预训练 (AffectNet / FER2013 / RAF-DB)
- 多任务学习 (AU 检测 + 表情分类联合训练)
