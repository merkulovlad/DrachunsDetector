
from dataclasses import dataclass

@dataclass
class PoseConfig:
    # YOLOv8-Pose weights (downloaded automatically by ultralytics if missing)
    weights: str = "yolov8n-pose.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    device: str = "cpu"  # "cpu" or "cuda" 
    max_det: int = 50

@dataclass
class TrackConfig:
    max_age: int = 30        # frames to keep lost track
    min_hits: int = 5       # confirmations before reporting a track
    iou_threshold: float = 0.0  # Now works correctly with fixed logic

@dataclass
class FeatureConfig:
    seq_len: int = 30        # frames per clip window (1-2s @ 15-30fps)
    stride: int = 15         # overlap stride between clips
    kp_count: int = 17       # COCO-style keypoints
    smooth_sigma: float = 0.0  # no gaussian smoothing by default

@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 1e-3
    max_epochs: int = 10
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    num_classes: int = 2
    workers: int = 4
    # Device configs - separate for pose vs model training
    pose_device: str = "cpu"      # YOLO-Pose (CPU for compatibility)
    model_device: str = "mps"    # BiLSTM model (auto-detect best available)

@dataclass
class RuntimeConfig:
    display: bool = True
    alert_threshold: float = 0.8
    save_alert_clips: bool = True
    out_dir: str = "alerts"
