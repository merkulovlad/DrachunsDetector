"""
Model A: ST-GCN based violence detection with YOLO pose estimation.
Based on ST-GCN/ implementation.
"""
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from collections import defaultdict, deque
import threading
from queue import Queue, Empty, Full

import cv2
import numpy as np
import torch
import torch.nn as nn

# Add ST-GCN to path for imports
stgcn_path = Path(__file__).parent.parent.parent / "ST-GCN"
sys.path.insert(0, str(stgcn_path))

from config import PoseConfig, TrackConfig, FeatureConfig, RuntimeConfig
from step2_pose import PoseEstimator
from tracking import Tracker


CONF_THRESH = 0.2
MIN_SCALE = 1e-2
MAX_RADIUS = 2.0


def build_coco_adjacency(num_joints: int = 17):
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    A = torch.eye(num_joints)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
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


class STGCNClassifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=7, graph_nodes=17, base_channels=64, dropout=0.5):
        super().__init__()
        A = build_coco_adjacency(graph_nodes)
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


def build_skeleton_tensor(track_seq, seq_len, frame_shape):
    """Build normalized skeleton tensor from track sequence."""
    if not track_seq:
        return np.zeros((seq_len, 17, 7), dtype=np.float32)
    H, W = frame_shape
    seq = list(track_seq)[-seq_len:]
    if len(seq) < seq_len:
        pad_template = np.zeros((17, 3), dtype=np.float32)
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

    skeleton_aug = np.concatenate([coords, vel, acc, conf], axis=-1)
    return skeleton_aug


def pick_best_track(histories, min_len):
    """Pick the track with longest history and best confidence."""
    best_tid = None
    best_score = (-1, -1.0)
    for tid, hist in histories.items():
        if len(hist) < min_len:
            continue
        conf = np.mean([kp[:, 2].mean() for kp in hist])
        score = (len(hist), conf)
        if score > best_score:
            best_tid = tid
            best_score = score
    return best_tid


class ViolenceDetectorModelA:
    """
    ST-GCN based violence detector with YOLO pose estimation.
    
    This model:
    1. Detects persons using YOLO pose estimation
    2. Tracks persons across frames
    3. Builds skeleton sequences for each tracked person
    4. Classifies violence using ST-GCN on skeleton sequences
    """
    
    def __init__(
        self,
        clf_checkpoint: str,
        pose_weights: str = "yolov8n-pose.pt",
        device: str = "cpu",
        seq_len: int = 30,
        stride: int = 15,
        threshold: float = 0.8,
    ):
        """
        Initialize Model A.
        
        Args:
            clf_checkpoint: Path to trained ST-GCN checkpoint
            pose_weights: Path to YOLO pose weights
            device: Device for inference ('cpu' or 'cuda')
            seq_len: Sequence length for skeleton clips
            stride: Stride between clips
            threshold: Violence probability threshold
        """
        self.device = device
        self.seq_len = seq_len
        self.stride = stride
        self.threshold = threshold
        
        # Initialize pose estimator
        self.pose = PoseEstimator(
            weights=pose_weights,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            device=device,
            max_det=50,
        )
        
        # Initialize tracker
        self.tracker = Tracker(
            max_age=30,
            min_hits=5,
            iou_threshold=0.0,
        )
        
        # Load classifier
        self.classifier = self._load_classifier(clf_checkpoint)
        
        # State tracking
        self.track_histories = defaultdict(lambda: deque(maxlen=seq_len))
        self.track_last_seen = {}
        self.frame_idx = 0
        self.last_infer_idx = -1
        self.last_prob = 0.0
        self.frame_shape = None
        
    def _load_classifier(self, ckpt_path: str) -> nn.Module:
        """Load the ST-GCN classifier from checkpoint."""
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")
        
        state = torch.load(ckpt_path, map_location=self.device)
        metadata = {
            "num_classes": state.get("num_classes", 2),
            "in_channels": state.get("in_channels", 7),
            "base_channels": state.get("base_channels", 64),
            "dropout": state.get("dropout", 0.5),
        }
        
        model = STGCNClassifier(
            num_classes=metadata["num_classes"],
            in_channels=metadata["in_channels"],
            base_channels=metadata["base_channels"],
            dropout=metadata["dropout"],
        ).to(self.device)
        
        model.load_state_dict(state["model"])
        model.eval()
        return model
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, List]:
        """
        Process a single frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (annotated_frame, violence_probability, tracks)
        """
        self.frame_idx += 1
        
        # Detect poses
        detections = self.pose.infer(frame)
        
        # Track persons
        tracks = self.tracker.step(detections)
        self.frame_shape = frame.shape[:2]
        
        # Update track histories
        for tid, bbox, keypoints in tracks:
            self.track_histories[tid].append(keypoints.copy())
            self.track_last_seen[tid] = self.frame_idx
        
        # Remove stale tracks
        stale_ids = [
            tid for tid, last_seen in self.track_last_seen.items()
            if self.frame_idx - last_seen > self.seq_len
        ]
        for tid in stale_ids:
            self.track_histories.pop(tid, None)
            self.track_last_seen.pop(tid, None)
        
        # Run inference if conditions are met
        min_len = max(4, self.seq_len // 2)
        if (
            self.frame_shape is not None
            and self.frame_idx >= min_len
            and (self.frame_idx - min_len) % self.stride == 0
            and self.frame_idx != self.last_infer_idx
        ):
            best_tid = pick_best_track(self.track_histories, min_len)
            if best_tid is not None:
                skeleton = build_skeleton_tensor(
                    list(self.track_histories[best_tid]),
                    self.seq_len,
                    self.frame_shape,
                )
                skeleton = np.transpose(skeleton, (2, 0, 1)).astype(np.float32, copy=False)
                
                # Run classifier
                tensor = torch.from_numpy(skeleton).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.classifier(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    self.last_prob = float(probs[1])
                
                self.last_infer_idx = self.frame_idx
        
        # Annotate frame
        annotated = self._annotate_frame(frame.copy(), tracks, self.last_prob)
        
        return annotated, self.last_prob, tracks
    
    def _annotate_frame(self, frame: np.ndarray, tracks: List, prob: float) -> np.ndarray:
        """Draw bounding boxes, keypoints and violence probability on frame."""
        # Violence probability text
        color = (0, 0, 255) if prob >= self.threshold else (0, 255, 0)
        prob_text = f"Violence prob: {prob:.2f}"
        cv2.putText(frame, prob_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Draw tracks
        for tid, bbox, keypoints in tracks:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw keypoints
            for x, y, conf in keypoints:
                if conf > 0.1:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
            
            # Track ID
            cv2.putText(
                frame, 
                f"ID {tid}", 
                (x1, y1 - 6), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                1
            )
        
        return frame
    
    def reset(self):
        """Reset detector state."""
        self.track_histories.clear()
        self.track_last_seen.clear()
        self.frame_idx = 0
        self.last_infer_idx = -1
        self.last_prob = 0.0
        self.frame_shape = None
        
        # Reset tracker
        self.tracker.tracks.clear()
