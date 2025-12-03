import argparse
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import FeatureConfig, PoseConfig, RuntimeConfig, TrackConfig
from step1_input import FrameSource
from step2_pose import PoseEstimator
from tracking import Tracker


CONF_THRESH = 0.2
MIN_SCALE = 1e-2
MAX_RADIUS = 2.0
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def build_multi_adjacency(num_persons: int, joints_per_person: int = 17):
    """Block-diagonal COCO adjacency + hip connections between persons."""
    V = num_persons * joints_per_person
    A = torch.eye(V)
    for p in range(num_persons):
        offset = p * joints_per_person
        for i, j in COCO_EDGES:
            A[offset + i, offset + j] = 1
            A[offset + j, offset + i] = 1
    hips = [11, 12]
    for pa in range(num_persons):
        for pb in range(pa + 1, num_persons):
            oa = pa * joints_per_person
            ob = pb * joints_per_person
            for h in hips:
                A[oa + h, ob + h] = 1
                A[ob + h, oa + h] = 1
    D = torch.diag(1.0 / torch.clamp(A.sum(dim=1), min=1.0))
    return D @ A


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, dropout=0.0):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = torch.einsum("nctv,vw->nctw", x, self.A)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x + res


class MultiPersonSTGCN(nn.Module):
    def __init__(self, num_classes=2, in_channels=7, num_persons=3, graph_nodes=None, base_channels=64, dropout=0.5):
        super().__init__()
        if graph_nodes is None:
            graph_nodes = num_persons * 17
        A = build_multi_adjacency(num_persons=num_persons, joints_per_person=graph_nodes // num_persons)
        self.register_buffer("A", A)
        channels = [base_channels, base_channels, base_channels, 2 * base_channels, 2 * base_channels, 4 * base_channels]
        strides = [1, 1, 1, 2, 1, 2]
        layers = []
        c_in = in_channels
        for c_out, s in zip(channels, strides):
            layers.append(GraphConv(c_in, c_out, self.A, stride=s, residual=True, dropout=dropout))
            c_in = c_out
        self.stgcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stgcn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_classifier(ckpt_path: str | None, device: str):
    if ckpt_path is None:
        return None, {}
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    metadata = {
        "num_classes": state.get("num_classes", 2),
        "in_channels": state.get("in_channels", 7),
        "graph_nodes": state.get("graph_nodes"),
        "num_persons": state.get("num_persons", 3),
        "base_channels": state.get("base_channels", 64),
        "dropout": state.get("dropout", 0.5),
    }
    model = MultiPersonSTGCN(
        num_classes=metadata["num_classes"],
        in_channels=metadata["in_channels"],
        num_persons=metadata["num_persons"],
        graph_nodes=metadata["graph_nodes"],
        base_channels=metadata["base_channels"],
        dropout=metadata["dropout"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, metadata


def _normalize_track(track_seq: list[np.ndarray], frame_shape, seq_len: int) -> np.ndarray:
    """Pad/trim a track and build 7-channel skeleton (coords/vel/acc/conf)."""
    H, W = frame_shape
    seq = list(track_seq)[-seq_len:]
    if len(seq) < seq_len:
        pad_template = np.zeros_like(seq[0]) if seq else np.zeros((17, 3), dtype=np.float32)
        seq = seq + [pad_template.copy() for _ in range(seq_len - len(seq))]
    skeleton = np.stack(seq, axis=0).astype(np.float32)
    skeleton[..., 0] /= max(W, 1e-6)
    skeleton[..., 1] /= max(H, 1e-6)

    coords = skeleton[..., :2]
    conf = skeleton[..., 2:3]

    low_conf_mask = np.repeat(conf < CONF_THRESH, 2, axis=2)
    coords[low_conf_mask] = 0.0

    hip_center = coords[:, [11, 12], :].mean(axis=1, keepdims=True)
    coords -= hip_center

    shoulder = coords[:, [5, 6], :]
    per_frame_len = np.linalg.norm(shoulder[:, 0] - shoulder[:, 1], axis=-1)
    valid = per_frame_len > MIN_SCALE
    scale = float(per_frame_len[valid].mean()) if np.any(valid) else 1.0
    coords /= max(scale, MIN_SCALE)

    radius_mask = np.repeat(np.linalg.norm(coords, axis=-1, keepdims=True) > MAX_RADIUS, 2, axis=2)
    coords[radius_mask] = 0.0

    vel = np.zeros_like(coords)
    vel[1:] = coords[1:] - coords[:-1]
    acc = np.zeros_like(coords)
    acc[2:] = vel[2:] - vel[1:-1]

    skeleton_aug = np.concatenate([coords, vel, acc, conf], axis=-1)  # (T, 17, 7)
    return skeleton_aug


def build_multi_skeleton_tensor(track_histories, seq_len, frame_shape, max_persons, min_len):
    candidates = []
    for tid, hist in track_histories.items():
        if len(hist) < min_len:
            continue
        conf = np.mean([kp[:, 2].mean() for kp in hist])
        candidates.append((len(hist), conf, tid, hist))
    if not candidates:
        return np.zeros((7, seq_len, max_persons * 17), dtype=np.float32)
    candidates.sort(reverse=True)
    selected = candidates[:max_persons]
    tracks_norm = []
    for _, _, _, hist in selected:
        tracks_norm.append(_normalize_track(list(hist), frame_shape, seq_len))
    while len(tracks_norm) < max_persons:
        tracks_norm.append(np.zeros((seq_len, 17, 7), dtype=np.float32))
    stack = np.stack(tracks_norm, axis=0)  # (P, T, 17, 7)
    stack = np.transpose(stack, (0, 3, 1, 2))  # (P, C, T, V)
    stack = stack.transpose(1, 2, 0, 3)  # (C, T, P, V)
    stack = stack.reshape(7, seq_len, max_persons * 17)
    return stack.astype(np.float32, copy=False)


def inference_worker(work_queue: Queue, result_state: dict, lock: threading.Lock, clf: nn.Module, device: str):
    while True:
        item = work_queue.get()
        if item is None:
            work_queue.task_done()
            break
        idx, skeleton = item
        tensor = torch.from_numpy(skeleton).unsqueeze(0).to(device)  # (1, C, T, V)
        with torch.no_grad():
            logits = clf(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        with lock:
            result_state["window_id"] = idx
            result_state["prob"] = float(probs[1])
        work_queue.task_done()


def detection_worker(frame_queue: Queue, result_queue: Queue, pose: PoseEstimator, tracker: Tracker, max_persons: int):
    while True:
        item = frame_queue.get()
        if item is None:
            frame_queue.task_done()
            break
        frame_idx, frame = item
        detections = pose.infer(frame)
        detections = sorted(detections, key=lambda d: d[1], reverse=True)[:max_persons]
        tracks = tracker.step(detections)
        result_queue.put((frame_idx, tracks, frame.shape[:2]))
        frame_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Async ST-GCN violence inference demo (multi-person).")
    parser.add_argument("--video", required=True, help="Path to video file.")
    parser.add_argument("--weights", default="yolov8n-pose.pt", help="Pose weights.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for models.")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--clf", required=True, help="Path to trained ST-GCN checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--write-out", default=None, help="Optional annotated video path.")
    parser.add_argument("--queue-size", type=int, default=4, help="Skeleton inference queue size.")
    parser.add_argument("--det-queue-size", type=int, default=3, help="Frame queue size for detection worker.")
    parser.add_argument("--max-persons", type=int, default=3, help="Max people to keep per frame/window.")
    args = parser.parse_args()

    video_arg = args.video
    if video_arg.isdigit():
        video_path = int(video_arg)  # webcam index
    else:
        video_path = Path(video_arg)
        if not video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path}")

    pose_cfg = PoseConfig(weights=args.weights, device=args.device)
    track_cfg = TrackConfig()
    feat_cfg = FeatureConfig(seq_len=args.seq_len, stride=args.stride)
    run_cfg = RuntimeConfig(alert_threshold=args.threshold)

    fs = FrameSource(video_path)
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
    clf, metadata = load_classifier(args.clf, args.device)
    if clf is None:
        raise ValueError("Classifier checkpoint must be provided for async demo.")

    fps = fs.cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_writer = None
    if args.write_out:
        out_path = Path(args.write_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cap = fs.cap
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    work_queue: Queue = Queue(maxsize=args.queue_size)
    result_state = {"prob": 0.0, "window_id": -1}
    state_lock = threading.Lock()
    worker = threading.Thread(
        target=inference_worker,
        args=(work_queue, result_state, state_lock, clf, args.device),
        daemon=True,
    )
    worker.start()

    frame_queue: Queue = Queue(maxsize=args.det_queue_size)
    result_queue: Queue = Queue(maxsize=args.det_queue_size * 2)
    det_thread = threading.Thread(
        target=detection_worker,
        args=(frame_queue, result_queue, pose, tracker, args.max_persons),
        daemon=True,
    )
    det_thread.start()

    prob_text = "Violence prob: --"
    last_prob = 0.0
    last_display = None
    track_histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=feat_cfg.seq_len))
    track_last_seen: dict[int, int] = {}
    frame_idx = 0
    processed_idx = 0
    last_infer_idx = -1
    window_counter = 0
    min_len = max(4, feat_cfg.seq_len // 2)
    latest_tracks = []
    frame_shape = None

    try:
        while True:
            ok, frame = fs.read()
            if not ok:
                break
            frame_idx += 1
            try:
                frame_queue.put_nowait((frame_idx, frame.copy()))
            except Full:
                pass

            while True:
                try:
                    proc_idx, tracks, det_shape = result_queue.get_nowait()
                except Empty:
                    break
                frame_shape = det_shape
                processed_idx = proc_idx
                for tid, bbox, keypoints in tracks:
                    track_histories[tid].append(keypoints.copy())
                    track_last_seen[tid] = proc_idx
                stale_ids = [tid for tid, last_seen in track_last_seen.items() if proc_idx - last_seen > feat_cfg.seq_len]
                for tid in stale_ids:
                    track_histories.pop(tid, None)
                    track_last_seen.pop(tid, None)

                latest_tracks = tracks

                if (
                    frame_shape is not None
                    and not work_queue.full()
                    and proc_idx >= min_len
                    and (proc_idx - min_len) % feat_cfg.stride == 0
                    and proc_idx != last_infer_idx
                ):
                    skeleton = build_multi_skeleton_tensor(
                        track_histories,
                        feat_cfg.seq_len,
                        frame_shape,
                        max_persons=args.max_persons,
                        min_len=min_len,
                    )
                    work_queue.put((window_counter, skeleton))
                    window_counter += 1
                    last_infer_idx = proc_idx

            with state_lock:
                if result_state["window_id"] >= 0:
                    last_prob = result_state["prob"]
                    prob_text = f"Violence prob: {last_prob:.2f}"

            now = time.perf_counter()
            if last_display is not None:
                elapsed = now - last_display
                if fps > 0:
                    target_period = 1.0 / fps
                    if elapsed < target_period:
                        time.sleep(target_period - elapsed)
                        now = time.perf_counter()
                fps_text = f"FPS: {1.0/elapsed:.1f}" if elapsed > 0 else "FPS: --"
            else:
                fps_text = "FPS: --"
            last_display = now

            color = (0, 0, 255) if last_prob >= run_cfg.alert_threshold else (0, 255, 0)
            cv2.putText(frame, prob_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, fps_text, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            for tid, bbox, keypoints in latest_tracks:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for x, y, conf in keypoints:
                    if conf > 0.1:
                        cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Async Violence Demo (Multi-person)", frame)
            if out_writer is not None:
                out_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        frame_queue.put(None)
        work_queue.put(None)
        det_thread.join()
        worker.join()
        if out_writer is not None:
            out_writer.release()
        fs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
