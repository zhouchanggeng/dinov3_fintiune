# 微笑识别 — DINOv3 + InsightFace

基于 Meta DINOv3 自监督视觉大模型的微笑/非微笑二分类表情识别系统。

## 核心流程

```
原图 → InsightFace 人脸检测对齐 → DINOv3 特征提取 → 分类头 → 微笑/非微笑
```

## 模型

使用本地预训练权重 `dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, 300M 参数, embed_dim=1024)，
通过 HuggingFace Transformers `AutoModel` 加载，提取 `pooler_output` (CLS token) 作为全局特征。

## 环境安装

```bash
pip install -r requirements.txt
```

> 注意: DINOv3 需要 `transformers>=4.56.0`

## 数据准备 (GENKI-4K)

1. 下载 GENKI-4K 数据集，解压至 `./data/GENKI-4K/`
2. 确保目录结构:
   ```
   data/GENKI-4K/
     files/          # file0001.jpg ~ file4000.jpg
     labels.txt      # 每行: label yaw pitch roll
   ```
3. 执行人脸检测、对齐与数据集划分:
   ```bash
   python prepare_data.py
   ```

## 训练

三种微调策略可选:

```bash
# 推荐: LoRA 参数高效微调 (兼顾精度与显存)
python train.py --mode lora

# 快速验证: 线性探测 (仅训练分类头)
python train.py --mode linear_probe

# 极致精度: 全量微调 (需要较大显存)
python train.py --mode full_finetune --lr 5e-5
```

## 推理

```bash
# 单张图片推理
python inference.py path/to/image.jpg --checkpoint ./outputs/best_model.pth

# 带可视化
python inference.py path/to/image.jpg --save result.jpg
```

## 微调策略对比

| 策略 | 可训练参数 | 显存占用 | 精度 | 适用场景 |
|------|-----------|---------|------|---------|
| Linear Probe | ~0.3M | 低 | ★★★ | 快速验证 |
| LoRA (r=8) | ~2M | 中 | ★★★★ | **推荐** |
| Full Finetune | ~300M | 高 | ★★★★★ | 极致精度 |

## 进阶优化方向

- 人脸精准仿射对齐 (3D 姿态估计 + 正面化)
- 人脸专用数据增强 (遮挡模拟、表情混合)
- 大规模人脸数据集域适配预训练 (AffectNet / FER2013 / RAF-DB)
- 多任务学习 (AU 检测 + 表情分类联合训练)
