"""
端到端推理 — 单张图片微笑分类
流程: 原图 -> InsightFace 人脸检测对齐 -> DINOv3 特征提取 -> 分类预测
"""
import argparse
import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

from config import Config
from model import build_model
from train import load_checkpoint
from prepare_data import face_align_by_landmarks
from dataset import get_val_transform


class SmilePredictor:
    """端到端微笑分类推理器"""

    def __init__(
        self,
        checkpoint_path: str,
        cfg: Config = None,
        device: str = "auto",
    ):
        self.cfg = cfg or Config()

        # 设备
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # 加载模型
        self.model = build_model(self.cfg).to(self.device)
        load_checkpoint(self.model, checkpoint_path)
        self.model.eval()

        # 图像变换 (与验证集一致)
        self.transform = get_val_transform(self.cfg)

        # InsightFace 人脸检测器
        self.face_app = None  # 延迟初始化

    def _init_face_detector(self):
        """延迟初始化 InsightFace"""
        from insightface.app import FaceAnalysis

        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_app.prepare(
            ctx_id=0,
            det_size=(self.cfg.face_det_size, self.cfg.face_det_size),
        )

    def detect_and_align(self, image: np.ndarray) -> np.ndarray:
        """人脸检测与仿射对齐"""
        if self.face_app is None:
            self._init_face_detector()

        faces = self.face_app.get(image)
        if len(faces) == 0:
            # 未检测到人脸 — 直接 resize
            print("  ⚠ 未检测到人脸，使用原图 resize")
            return cv2.resize(
                image, (self.cfg.face_align_size, self.cfg.face_align_size)
            )

        # 取置信度最高的人脸
        face = max(faces, key=lambda f: f.det_score)
        aligned = face_align_by_landmarks(
            image, face.kps, output_size=self.cfg.face_align_size
        )
        return aligned

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        单张图片端到端推理。

        返回:
          {
            "label": 0 或 1,
            "class_name": "微笑" 或 "非微笑",
            "confidence": float,
            "probabilities": {"非微笑": float, "微笑": float},
          }
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        # 人脸检测与对齐
        aligned = self.detect_and_align(img)
        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        # 图像预处理 (DINOv3 归一化)
        augmented = self.transform(image=aligned_rgb)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        # 模型推理
        with autocast():
            logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        label = int(probs.argmax())
        class_names = ["非微笑", "微笑"]

        result = {
            "label": label,
            "class_name": class_names[label],
            "confidence": float(probs[label]),
            "probabilities": {
                "非微笑": float(probs[0]),
                "微笑": float(probs[1]),
            },
        }
        return result

    def predict_with_visualization(self, image_path: str, save_path: str = None):
        """推理并可视化结果"""
        result = self.predict(image_path)

        img = cv2.imread(image_path)
        # 在图片上绘制结果
        text = f"{result['class_name']} ({result['confidence']:.2%})"
        color = (0, 255, 0) if result["label"] == 1 else (0, 0, 255)
        cv2.putText(
            img, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
        )

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"  结果已保存至: {save_path}")

        return result, img


def main():
    parser = argparse.ArgumentParser(description="微笑分类推理")
    parser.add_argument("image", type=str, help="输入图片路径")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs/best_model.pth",
        help="模型检查点路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="lora",
        choices=["linear_probe", "lora", "full_finetune"],
    )
    parser.add_argument("--save", type=str, default=None, help="可视化结果保存路径")
    args = parser.parse_args()

    cfg = Config()
    cfg.finetune_mode = args.mode
    cfg.__post_init__()

    predictor = SmilePredictor(args.checkpoint, cfg)

    print(f"\n推理图片: {args.image}")
    print("-" * 40)

    if args.save:
        result, _ = predictor.predict_with_visualization(args.image, args.save)
    else:
        result = predictor.predict(args.image)

    print(f"  预测结果: {result['class_name']}")
    print(f"  置信度:   {result['confidence']:.4f}")
    print(f"  概率分布: 非微笑={result['probabilities']['非微笑']:.4f}, "
          f"微笑={result['probabilities']['微笑']:.4f}")


if __name__ == "__main__":
    main()
