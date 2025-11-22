
from dataclasses import dataclass

@dataclass
class PoseConfig:
    # YOLOv8-Pose weights (downloaded automatically by ultralytics if missing)
    weights: str = "yolov8n-pose.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    device: str = "cpu" 
    max_det: int = 50

@dataclass
class TrackConfig:
    max_age: int = 30        # frames to keep lost track
    min_hits: int = 5       # confirmations before reporting a track
    iou_threshold: float = 0.0  

@dataclass
class FeatureConfig:
    seq_len: int = 30        # frames per clip window (1-2s @ 15-30fps)
    stride: int = 15         # overlap stride between clips
    
@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    max_epochs: int = 10
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    num_classes: int = 2
    workers: int = 0
    model_device: str = "mps"  
    joint_dropout: float = 0.1

@dataclass
class RuntimeConfig:
    display: bool = True
    alert_threshold: float = 0.8
    save_alert_clips: bool = True
    out_dir: str = "alerts"
