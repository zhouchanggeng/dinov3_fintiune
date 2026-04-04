"""
GENKI-4K 数据集准备 — 人脸检测、对齐与标准化裁剪
使用 InsightFace 完成人脸检测与 5 点关键点仿射对齐，消除背景干扰。

本数据集结构 (kaggle-genki4k):
  data/kaggle-genki4k/
    smile/        # 微笑图片
    non_smile/    # 非微笑图片

用法:
  python prepare_data.py
"""
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

from config import Config


def load_samples(data_root: str) -> list[tuple[str, int]]:
    """
    加载 kaggle-genki4k 数据集。
    目录结构: smile/ (label=1), non_smile/ (label=0)
    返回 [(image_path, label), ...]
    """
    samples = []
    label_map = {"non_smile": 0, "smile": 1}

    for folder, label in label_map.items():
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            print(f"  ⚠ 目录不存在: {folder_path}")
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(folder_path, fname), label))

    return samples


def align_faces(cfg: Config):
    """
    对全部图片执行人脸检测与仿射对齐。
    对齐后的人脸图片按 label 存储:
      aligned_dir/0/fileXXXX.jpg  (非微笑)
      aligned_dir/1/fileXXXX.jpg  (微笑)
    """
    # 初始化 InsightFace
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(cfg.face_det_size, cfg.face_det_size))

    samples = load_samples(cfg.data_root)
    print(f"共加载 {len(samples)} 个样本")

    # 创建输出目录
    for label in [0, 1]:
        os.makedirs(os.path.join(cfg.aligned_dir, str(label)), exist_ok=True)

    success, fail = 0, 0
    for img_path, label in tqdm(samples, desc="人脸对齐"):
        img = cv2.imread(img_path)
        if img is None:
            fail += 1
            continue

        faces = app.get(img)
        if len(faces) == 0:
            # 未检测到人脸 — 直接 resize 作为 fallback
            aligned = cv2.resize(img, (cfg.face_align_size, cfg.face_align_size))
        else:
            # 取置信度最高的人脸
            face = max(faces, key=lambda f: f.det_score)
            aligned = face_align_by_landmarks(
                img, face.kps, output_size=cfg.face_align_size
            )

        # 保存
        fname = os.path.basename(img_path)
        out_path = os.path.join(cfg.aligned_dir, str(label), fname)
        cv2.imwrite(out_path, aligned)
        success += 1

    print(f"对齐完成: 成功 {success}, 失败 {fail}")


def face_align_by_landmarks(
    img: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """
    基于 5 点人脸关键点的精准仿射对齐。
    参考标准模板 (arcface_dst) 进行相似变换，
    将人脸归一化到统一的正面姿态。
    """
    # ArcFace 标准 5 点模板 (112x112)
    arcface_dst = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    # 缩放模板到目标尺寸
    scale = output_size / 112.0
    dst = arcface_dst * scale

    # 估计相似变换矩阵
    src_pts = landmarks.astype(np.float32)
    tform = cv2.estimateAffinePartial2D(src_pts, dst)[0]

    if tform is None:
        return cv2.resize(img, (output_size, output_size))

    aligned = cv2.warpAffine(
        img, tform, (output_size, output_size), borderValue=0.0
    )
    return aligned


def split_dataset(cfg: Config):
    """
    将对齐后的数据集按比例划分为 train/val/test，
    生成对应的文件列表 (txt)。
    """
    from sklearn.model_selection import train_test_split

    all_samples = []
    for label in [0, 1]:
        label_dir = os.path.join(cfg.aligned_dir, str(label))
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_samples.append((os.path.join(label_dir, fname), label))

    print(f"对齐数据集共 {len(all_samples)} 个样本")

    # 分层划分
    paths, labels = zip(*all_samples)
    paths, labels = list(paths), list(labels)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(1 - cfg.train_ratio),
        stratify=labels,
        random_state=cfg.seed,
    )
    relative_val = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - relative_val),
        stratify=temp_labels,
        random_state=cfg.seed,
    )

    # 写入文件列表
    for split_name, s_paths, s_labels in [
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels),
    ]:
        out_file = os.path.join(cfg.aligned_dir, f"{split_name}.txt")
        with open(out_file, "w") as f:
            for p, l in zip(s_paths, s_labels):
                f.write(f"{p}\t{l}\n")
        print(f"  {split_name}: {len(s_paths)} 样本 -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GENKI-4K 数据准备与人脸对齐")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--aligned_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.data_root = args.data_root
    if args.aligned_dir:
        cfg.aligned_dir = args.aligned_dir
    cfg.__post_init__()

    print("=" * 50)
    print("Step 1: 人脸检测与仿射对齐")
    print("=" * 50)
    align_faces(cfg)

    print("\n" + "=" * 50)
    print("Step 2: 数据集划分 (train/val/test)")
    print("=" * 50)
    split_dataset(cfg)
