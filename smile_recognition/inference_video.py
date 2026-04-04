"""视频微笑识别推理 — 逐帧检测所有人脸并标注表情"""
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


def main():
    parser = argparse.ArgumentParser(description="视频微笑识别")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("-o", "--output", default="output_video.mp4", help="输出视频路径")
    parser.add_argument("--checkpoint", default="./outputs/best_model.pth")
    parser.add_argument("--mode", default="lora", choices=["linear_probe", "lora", "full_finetune"])
    args = parser.parse_args()

    cfg = Config()
    cfg.finetune_mode = args.mode
    cfg.__post_init__()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = build_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    transform = get_val_transform(cfg)

    # 初始化人脸检测器
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(cfg.face_det_size, cfg.face_det_size))

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    class_names = ["normal", "smile"]
    frame_idx = 0

    print(f"视频: {w}x{h}, {fps:.1f}fps, {total}帧")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            faces = face_app.get(frame)

            for face in faces:
                # 对齐人脸并推理
                aligned = face_align_by_landmarks(frame, face.kps, output_size=cfg.face_align_size)
                aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                tensor = transform(image=aligned_rgb)["image"].unsqueeze(0).to(device)

                with autocast():
                    logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                label = int(probs.argmax())
                conf = float(probs[label])

                # 画框和标签
                x1, y1, x2, y2 = face.bbox.astype(int)
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text = f"{class_names[label]} {conf:.0%}"
                # 背景框让文字更清晰
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out.write(frame)

            if frame_idx % 30 == 0 or frame_idx == total:
                print(f"\r处理进度: {frame_idx}/{total} ({frame_idx*100//total}%)", end="", flush=True)

    cap.release()
    out.release()
    print(f"\n完成! 输出: {args.output}")


if __name__ == "__main__":
    main()
