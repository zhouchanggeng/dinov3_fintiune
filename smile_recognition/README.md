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
├── prepare_data.py        # 人脸检测、对齐与数据集划分 (支持 GENKI-4K + RAF-DB)
├── dataset.py             # Dataset & 数据增强
├── model.py               # DINOv3 backbone + 分类头 (Linear Probe / LoRA / Full)
├── train.py               # 训练循环 (含早停、学习率调度、多GPU、SwanLab)
├── evaluate.py            # 评估指标 & 可视化
├── inference.py           # 单张图片端到端推理
├── inference_video.py     # 视频推理
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

支持两个数据集，可单独使用也可合并训练：

### 数据集 1: GENKI-4K

Kaggle 版 GENKI-4K，已按类别分目录：

```
data/kaggle-genki4k/
  smile/          # 2162 张微笑图片
  non_smile/      # 1838 张非微笑图片
```

处理流程：InsightFace (buffalo_l) 人脸检测 → 5 点关键点仿射对齐 (ArcFace 标准模板) → 统一裁剪至 112×112。

### 数据集 2: RAF-DB

RAF-DB 包含 7 类表情 (1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral)，
图片已预对齐。自动映射为二分类：label 4 (Happiness) → 微笑，其余 → 非微笑。

```
data/RAF-DB/
  DATASET/
    train/{1..7}/    # 12271 张训练图片
    test/{1..7}/     # 3068 张测试图片
  train_labels.csv
  test_labels.csv
```

### 执行数据准备

```bash
# 同时处理 GENKI-4K + RAF-DB (默认)
python prepare_data.py

# 仅处理 GENKI-4K
python prepare_data.py --skip_rafdb

# 仅导入 RAF-DB
python prepare_data.py --skip_genki

# 自定义路径
python prepare_data.py --data_root ./data/kaggle-genki4k --rafdb_root ./data/RAF-DB --aligned_dir ./data/aligned
```

处理完成后统一存储至 `data/GENKI-4K_aligned/{0,1}/`，并自动分层划分 train/val/test (80%/10%/10%)。

## 训练

三种微调策略可选：

```bash
# 推荐: LoRA 参数高效微调 (兼顾精度与显存)
CUDA_VISIBLE_DEVICES=0,2,4,6 python train.py --mode lora

# 快速验证: 线性探测 (仅训练分类头)
CUDA_VISIBLE_DEVICES=0 python train.py --mode linear_probe

# 极致精度: 全量微调 (需要较大显存)
CUDA_VISIBLE_DEVICES=0,2,4,6 python train.py --mode full_finetune --lr 5e-5
```

训练特性：
- Warmup + Cosine Annealing 学习率调度
- 混合精度训练 (AMP)
- 梯度裁剪 (max_norm=1.0)
- 早停机制 (patience=7)
- Label Smoothing (0.1)
- 多 GPU DataParallel 支持
- SwanLab 实验追踪

## 推理

```bash
# 单张图片推理
python inference.py path/to/image.jpg --checkpoint ./outputs/best_model.pth

# 带可视化输出
python inference.py path/to/image.jpg --save result.jpg
```

## 评估指标

训练完成后自动在测试集上输出：
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
- 更多大规模人脸数据集域适配 (AffectNet / FER2013)
- 多任务学习 (AU 检测 + 表情分类联合训练)
