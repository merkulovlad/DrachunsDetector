import argparse
import time
from pathlib import Path

import cv2
import torch

from config import FeatureConfig, PoseConfig, RuntimeConfig, TrackConfig
from features import aggregate_track_window, pool_across_tracks
from model import BiLSTMClassifier
from step1_input import FrameSource
from step2_pose import PoseEstimator
from tracking import Tracker


def load_classifier(ckpt_path: str | None, device: str):
    """Load a trained BiLSTM checkpoint if provided."""
    if ckpt_path is None:
        return None, None
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model = BiLSTMClassifier(
        input_dim=state["Fdim"],
        hidden_size=state["hidden"],
        num_layers=state["layers"],
        dropout=state["dropout"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["Fdim"]


def main():
    parser = argparse.ArgumentParser(description="Run violence probability overlay on a video file.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--weights", default="yolov8n-pose.pt", help="Pose weights to use.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for pose + classifier."
    )
    parser.add_argument("--seq_len", type=int, default=30, help="Sliding window length.")
    parser.add_argument("--stride", type=int, default=15, help="Sliding window stride.")
    parser.add_argument("--clf", default=None, help="Path to a trained classifier checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Alert threshold for highlighting.")
    parser.add_argument("--write-out", default=None, help="Optional path to save annotated video.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    pose_cfg = PoseConfig(weights=args.weights, device=args.device)
    track_cfg = TrackConfig()
    feat_cfg = FeatureConfig(seq_len=args.seq_len, stride=args.stride)
    run_cfg = RuntimeConfig(alert_threshold=args.threshold)

    fs = FrameSource(str(video_path))
    pose = PoseEstimator(
        weights=pose_cfg.weights,
        imgsz=pose_cfg.imgsz,
        conf=pose_cfg.conf,
        iou=pose_cfg.iou,
        device=pose_cfg.device,
        max_det=pose_cfg.max_det,
    )
    tracker = Tracker(
        max_age=track_cfg.max_age,
        min_hits=track_cfg.min_hits,
        iou_threshold=track_cfg.iou_threshold,
    )
    clf, _ = load_classifier(args.clf, args.device)

    out_writer = None
    if args.write_out:
        out_path = Path(args.write_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cap = fs.cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    win_tracks = []
    prob_text = "Violence prob: --"
    last_prob = 0.0
    last_time = None

    while True:
        ok, frame = fs.read()
        if not ok:
            break

        detections = pose.infer(frame)
        tracks = tracker.step(detections)

        for tid, bbox, keypoints in tracks:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x, y, conf in keypoints:
                if conf > 0.1:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        win_tracks.append([tuple(t) for t in tracks])
        if len(win_tracks) >= feat_cfg.seq_len:
            track_kps = {}
            for frame_tracks in win_tracks:
                for tid, bbox, kp in frame_tracks:
                    track_kps.setdefault(tid, []).append(kp)
            track_feats = []
            for seq in track_kps.values():
                if len(seq) >= max(2, feat_cfg.seq_len // 2):
                    track_feats.append(aggregate_track_window(seq))
            pooled = pool_across_tracks(track_feats)
            if clf is not None:
                X = torch.from_numpy(pooled).float().unsqueeze(0).unsqueeze(1).to(args.device)
                with torch.no_grad():
                    logits = clf(X)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                last_prob = float(probs[1])
                prob_text = f"Violence prob: {last_prob:.2f}"
            win_tracks = win_tracks[feat_cfg.stride :]

        now = time.time()
        if last_time is not None:
            elapsed = now - last_time
            fps_text = f"FPS: {1.0/elapsed:.1f}" if elapsed > 0 else "FPS: --"
        else:
            fps_text = "FPS: --"
        last_time = now

        color = (0, 0, 255) if last_prob >= run_cfg.alert_threshold else (0, 255, 0)
        cv2.putText(frame, prob_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, fps_text, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Violence Probability Demo", frame)
        if out_writer is not None:
            out_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if out_writer is not None:
        out_writer.release()
    fs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
