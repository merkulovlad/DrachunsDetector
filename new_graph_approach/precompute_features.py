import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config import FeatureConfig, PoseConfig, TrackConfig
from features import aggregate_track_window, pool_across_tracks
from step2_pose import PoseEstimator
from tracking import Tracker

FEATURE_DIM = 96  # Expected pooled feature length per window

_WORKER_CONTEXT: Dict[str, object] = {}


def _init_worker(pose_cfg_dict: Dict, feature_cfg_dict: Dict, track_cfg_dict: Dict, max_persons: int):
    """Initializer for worker processes so the YOLO weights are loaded once per worker."""
    global _WORKER_CONTEXT
    pose_cfg = PoseConfig(**pose_cfg_dict)
    feature_cfg = FeatureConfig(**feature_cfg_dict)
    track_cfg = TrackConfig(**track_cfg_dict)
    pose_estimator = PoseEstimator(
        weights=pose_cfg.weights,
        imgsz=pose_cfg.imgsz,
        conf=pose_cfg.conf,
        iou=pose_cfg.iou,
        device=pose_cfg.device,
        max_det=pose_cfg.max_det,
    )
    _WORKER_CONTEXT = {
        "pose": pose_estimator,
        "feature_cfg": feature_cfg,
        "track_cfg": track_cfg,
        "max_persons": max_persons,
    }


def _worker_process(task: Tuple[str, str, int, str, str, str]):
    split, label_name, label, video_path_str, out_path_str, rel_output = task
    ctx = _WORKER_CONTEXT
    video_path = Path(video_path_str)
    out_path = Path(out_path_str)
    features, labels, stats = process_video(
        video_path=video_path,
        label=label,
        pose=ctx["pose"],
        feature_cfg=ctx["feature_cfg"],
        track_cfg=ctx["track_cfg"],
        max_persons=ctx["max_persons"],
    )
    np.savez_compressed(out_path, features=features, labels=labels)
    return {
        "split": split,
        "label_name": label_name,
        "label": label,
        "video": video_path.name,
        "output": rel_output,
        "num_frames": stats["num_frames"],
        "num_windows": stats["num_windows"],
    }


def read_video_frames(video_path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def generate_windows(num_frames: int, seq_len: int, stride: int) -> List[Tuple[int, int]]:
    if num_frames == 0:
        return []
    if num_frames <= seq_len:
        return [(0, num_frames)]
    windows: List[Tuple[int, int]] = []
    for start in range(0, num_frames - seq_len + 1, stride):
        end = start + seq_len
        windows.append((start, end))
    if not windows:
        windows.append((num_frames - seq_len, num_frames))
    return windows


def window_features(
    frames: List[np.ndarray],
    pose: PoseEstimator,
    track_cfg: TrackConfig,
    max_persons: int,
    feature_dim: int = FEATURE_DIM,
) -> np.ndarray:
    tracker = Tracker(
        max_age=track_cfg.max_age,
        min_hits=track_cfg.min_hits,
        iou_threshold=track_cfg.iou_threshold,
    )
    keypoints_by_track: Dict[int, List[np.ndarray]] = {}
    detections_per_frame = pose.infer_batch(frames)
    for detections in detections_per_frame:
        detections = sorted(detections, key=lambda d: d[1], reverse=True)[:max_persons]
        tracks = tracker.step(detections)
        for track_id, _, keypoints in tracks:
            keypoints_by_track.setdefault(track_id, []).append(keypoints)

    track_features = [
        aggregate_track_window(sequence)
        for sequence in keypoints_by_track.values()
        if len(sequence) >= max(2, len(frames) // 2)
    ]
    pooled = pool_across_tracks(track_features)
    if pooled.shape[0] != feature_dim:
        if pooled.shape[0] < feature_dim:
            pooled = np.pad(pooled, (0, feature_dim - pooled.shape[0]), mode="constant")
        else:
            pooled = pooled[:feature_dim]
    return pooled.astype(np.float32)


def process_video(
    video_path: Path,
    label: int,
    pose: PoseEstimator,
    feature_cfg: FeatureConfig,
    track_cfg: TrackConfig,
    max_persons: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    frames = read_video_frames(video_path)
    windows = generate_windows(len(frames), feature_cfg.seq_len, feature_cfg.stride)
    features: List[np.ndarray] = []
    for start, end in windows:
        window = frames[start:end]
        features.append(window_features(window, pose, track_cfg, max_persons))
    if not features:
        features.append(np.zeros((FEATURE_DIM,), dtype=np.float32))
    feature_array = np.stack(features, axis=0)
    labels = np.full((feature_array.shape[0],), label, dtype=np.int64)
    stats = {
        "num_frames": len(frames),
        "num_windows": feature_array.shape[0],
    }
    return feature_array, labels, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute pose-based features for RWF-2000 videos.")
    parser.add_argument("--data-root", required=True, help="Path to RWF-2000 dataset root.")
    parser.add_argument("--output-root", default="precomputed_features", help="Directory to store .npz feature files.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to process.")
    parser.add_argument("--max-persons", type=int, default=10, help="Top-N persons per frame to keep.")
    parser.add_argument("--pose-device", type=str, default=None, help="Override pose device (cpu/cuda/auto).")
    parser.add_argument("--pose-weights", type=str, default=None, help="Path to pose weights override.")
    parser.add_argument("--pose-imgsz", type=int, default=None, help="Override pose inference image size.")
    parser.add_argument("--pose-conf", type=float, default=None, help="Override pose confidence threshold.")
    parser.add_argument("--pose-iou", type=float, default=None, help="Override pose IoU threshold.")
    parser.add_argument("--pose-max-det", type=int, default=None, help="Override max detections per image.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override feature window length.")
    parser.add_argument("--stride", type=int, default=None, help="Override window stride.")
    parser.add_argument("--track-max-age", type=int, default=None, help="Override tracker max_age.")
    parser.add_argument("--track-min-hits", type=int, default=None, help="Override tracker min_hits.")
    parser.add_argument("--track-iou-threshold", type=float, default=None, help="Override tracker IoU threshold.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz outputs.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N videos per split (useful for smoke tests).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel worker processes (0 runs sequentially).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pose_cfg = PoseConfig()
    if args.pose_weights is not None:
        pose_cfg.weights = args.pose_weights
    if args.pose_imgsz is not None:
        pose_cfg.imgsz = args.pose_imgsz
    if args.pose_conf is not None:
        pose_cfg.conf = args.pose_conf
    if args.pose_iou is not None:
        pose_cfg.iou = args.pose_iou
    if args.pose_max_det is not None:
        pose_cfg.max_det = args.pose_max_det
    if args.pose_device is not None:
        pose_cfg.device = args.pose_device

    feature_cfg = FeatureConfig()
    if args.seq_len is not None:
        feature_cfg.seq_len = args.seq_len
    if args.stride is not None:
        feature_cfg.stride = args.stride

    track_cfg = TrackConfig()
    if args.track_max_age is not None:
        track_cfg.max_age = args.track_max_age
    if args.track_min_hits is not None:
        track_cfg.min_hits = args.track_min_hits
    if args.track_iou_threshold is not None:
        track_cfg.iou_threshold = args.track_iou_threshold

    manifest: List[Dict[str, object]] = []
    pose_cfg_dict = asdict(pose_cfg)
    feature_cfg_dict = asdict(feature_cfg)
    track_cfg_dict = asdict(track_cfg)
    label_pairs = [("Fight", 1), ("NonFight", 0)]

    processed_total = 0
    skipped_total = 0
    tasks: List[Tuple[str, str, int, str, str, str]] = []

    for split in args.splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"[WARN] Split directory missing: {split_dir}, skipping.")
            continue

        for label_name, label in label_pairs:
            label_dir = split_dir / label_name
            if not label_dir.exists():
                print(f"[WARN] Label directory missing: {label_dir}, skipping.")
                continue

            videos = sorted(
                list(label_dir.glob("*.mp4")) + list(label_dir.glob("*.avi"))
            )
            if args.limit is not None:
                videos = videos[: args.limit]

            out_dir = output_root / split / label_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for video_path in videos:
                out_path = out_dir / f"{video_path.stem}.npz"
                rel_output = str(out_path.relative_to(output_root))

                if out_path.exists() and not args.overwrite:
                    skipped_total += 1
                    continue

                tasks.append(
                    (
                        split,
                        label_name,
                        label,
                        str(video_path),
                        str(out_path),
                        rel_output,
                    )
                )

    if tasks:
        if args.num_workers == 0:
            pose_estimator = PoseEstimator(
                weights=pose_cfg.weights,
                imgsz=pose_cfg.imgsz,
                conf=pose_cfg.conf,
                iou=pose_cfg.iou,
                device=pose_cfg.device,
                max_det=pose_cfg.max_det,
            )
            for task in tqdm(tasks, desc="Processing videos", unit="video"):
                split, label_name, label, video_path_str, out_path_str, rel_output = task
                video_path = Path(video_path_str)
                out_path = Path(out_path_str)
                features, labels, stats = process_video(
                    video_path=video_path,
                    label=label,
                    pose=pose_estimator,
                    feature_cfg=feature_cfg,
                    track_cfg=track_cfg,
                    max_persons=args.max_persons,
                )
                np.savez_compressed(out_path, features=features, labels=labels)
                manifest.append(
                    {
                        "split": split,
                        "label_name": label_name,
                        "label": label,
                        "video": video_path.name,
                        "output": rel_output,
                        "num_frames": stats["num_frames"],
                        "num_windows": stats["num_windows"],
                    }
                )
                processed_total += 1
        else:
            initargs = (pose_cfg_dict, feature_cfg_dict, track_cfg_dict, args.max_persons)
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                initializer=_init_worker,
                initargs=initargs,
            ) as executor:
                futures = {
                    executor.submit(_worker_process, task): task for task in tasks
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(tasks),
                    desc="Processing videos",
                    unit="video",
                ):
                    manifest_entry = future.result()
                    manifest.append(manifest_entry)
                    processed_total += 1

    meta = {
        "pose_cfg": pose_cfg_dict,
        "feature_cfg": feature_cfg_dict,
        "track_cfg": track_cfg_dict,
        "max_persons": args.max_persons,
        "manifest": manifest,
    }
    metadata_path = output_root / "manifest.json"
    metadata_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[DONE] Manifest written to {metadata_path} | processed {processed_total}, skipped {skipped_total}."
    )


if __name__ == "__main__":
    main()
