
import os, glob, cv2, numpy as np, torch
from torch.utils.data import Dataset
from step2_pose import PoseEstimator
from tracking import Tracker
from features import aggregate_track_window, pool_across_tracks
from config import PoseConfig, TrackConfig, FeatureConfig
from pathlib import Path

class RWF2000Clips(Dataset):
    def __init__(self, root, split="train", seq_len=30, stride=15,
                 pose_cfg:PoseConfig=None, track_cfg:TrackConfig=None):
        self.root = root
        self.split = split
        self.seq_len = seq_len
        self.stride = stride
        self.items = []  # list of (video_path, label)
        for label_name, y in [("Fight",1), ("NonFight",0)]:
            vid_dir = os.path.join(root, split, label_name)
            for vp in sorted(glob.glob(os.path.join(vid_dir, "*.mp4")) + glob.glob(os.path.join(vid_dir, "*.avi"))):
                self.items.append((vp, y))

        self.pose = PoseEstimator(weights=pose_cfg.weights, imgsz=pose_cfg.imgsz,
                                  conf=pose_cfg.conf, iou=pose_cfg.iou, device=pose_cfg.device, max_det=pose_cfg.max_det)
        self.track_cfg = track_cfg

    def __len__(self):
        return len(self.items)

    def _video_to_windows(self, vpath):
        cap = cv2.VideoCapture(vpath)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok: break
            frames.append(fr)
        cap.release()
        windows = []
        for start in range(0, max(1, len(frames)-self.seq_len+1), self.stride):
            end = start + self.seq_len
            if end <= len(frames):
                windows.append(frames[start:end])
        return windows

    def __getitem__(self, idx):
        vpath, y = self.items[idx]
        windows = self._video_to_windows(vpath)
        X = []
        for win in windows:
            tracker = Tracker(max_age=self.track_cfg.max_age,
                              min_hits=self.track_cfg.min_hits,
                              iou_threshold=self.track_cfg.iou_threshold)
            track_feats = []
            for fr in win:
                dets = self.pose.infer(fr)
                # Only keep top-k persons by score
                dets = sorted(dets, key=lambda d: d[1], reverse=True)[:10]
                tracks = tracker.step(dets)
                # Save KP by track id for this frame
                # We'll collect sequences at the end per track
                pass
            # Re-run tracking to capture kp sequences for this window
            tracker = Tracker(max_age=self.track_cfg.max_age,
                              min_hits=self.track_cfg.min_hits,
                              iou_threshold=self.track_cfg.iou_threshold)
            # Build dict: id -> list of kp
            kps_by_id = {}
            for fr in win:
                dets = self.pose.infer(fr)
                dets = sorted(dets, key=lambda d: d[1], reverse=True)[:10]
                tracks = tracker.step(dets)
                for tid, bbox, kp in tracks:
                    kps_by_id.setdefault(tid, []).append(kp)
            # Aggregate per-track (only those with full length)
            for tid, seq in kps_by_id.items():
                if len(seq) >= max(2, len(win)//2):  # keep reasonable
                    track_feats.append(aggregate_track_window(seq))
            pooled = pool_across_tracks(track_feats)
            X.append(pooled)
        if len(X)==0:
            # Create dummy feature with consistent dimension
            # per_frame_features returns: center(2) + vel(2) + acc(2) + [mean_speed, size_proxy](2) + angs(8) = 16
            # aggregate_track_window: 3 * 16 = 48
            # pool_across_tracks: 2 * 48 = 96
            dummy_feat = np.zeros((96,), dtype=np.float32)
            X = [dummy_feat]
        
        # Ensure all features have the same dimension
        target_dim = 96  # Expected feature dimension
        X_fixed = []
        for feat in X:
            if len(feat) != target_dim:
                # Pad or truncate to target dimension
                if len(feat) < target_dim:
                    feat = np.pad(feat, (0, target_dim - len(feat)), mode='constant')
                else:
                    feat = feat[:target_dim]
            X_fixed.append(feat)
        X = X_fixed
        
        X = np.stack(X, axis=0)  # (num_windows, Fagg)
        y = np.full((X.shape[0],), y, dtype=np.int64)
        return torch.from_numpy(X), torch.from_numpy(y)


class RWF2000Precomputed(Dataset):
    """
    Dataset that loads precomputed feature windows saved as .npz files.
    Expected directory layout:
        features_root/
            train/
                Fight/
                    video1.npz
                NonFight/
                    video2.npz
            val/
                ...
    Each npz file must contain an array `features` with shape (num_windows, feature_dim).
    """

    def __init__(self, features_root: str | os.PathLike, split: str = "train", feature_key: str = "features"):
        self.features_root = Path(features_root)
        self.split = split
        self.feature_key = feature_key
        split_dir = self.features_root / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Precomputed split directory not found: {split_dir}")

        self.items = []  # list of (path, label)
        for label_name, label in [("Fight", 1), ("NonFight", 0)]:
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            for npz_path in sorted(label_dir.glob("*.npz")):
                self.items.append((npz_path, label))

        if not self.items:
            raise RuntimeError(f"No precomputed features found in {split_dir}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        npz_path, label = self.items[index]
        with np.load(npz_path) as data:
            if self.feature_key not in data:
                raise KeyError(f"Feature key '{self.feature_key}' missing in {npz_path}")
            features = data[self.feature_key].astype(np.float32)
            if "labels" in data:
                labels = data["labels"].astype(np.int64)
            else:
                labels = np.full((features.shape[0],), label, dtype=np.int64)
        return torch.from_numpy(features), torch.from_numpy(labels)
