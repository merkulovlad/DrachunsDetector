"""
Model B: MoViNet-based streaming violence detection.
Based on streaming_platform/ implementation.
"""
import sys
from pathlib import Path
from typing import List, Optional
from collections import deque
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add streaming_platform to path for imports
stream_path = Path(__file__).parent.parent.parent / "stream_platform"
sys.path.insert(0, str(stream_path))

from backend.models.load_model import build_movinet_a0_stream


def preprocess_frame(bgr: np.ndarray) -> torch.Tensor:
    """Preprocess a single BGR frame for MoViNet."""
    import torchvision.transforms.v2 as T
    
    frame_tf = T.Compose([
        T.ToImage(),
        T.ConvertImageDtype(torch.float32),
        T.Resize(172),
        T.CenterCrop(172),
        T.Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
    ])
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = frame_tf(rgb) / 255.0
    return x


def prepare_clip_tensor(clip):
    """
    Ensure the input clip matches MoViNet expected shape: (B, C, T, H, W).
    Accepts a tensor or an iterable of frame tensors (C, H, W).
    """
    if isinstance(clip, torch.Tensor):
        clip_tensor = clip
    else:
        clip_tensor = torch.stack(list(clip), dim=0)  # T, C, H, W

    if clip_tensor.ndim == 4:
        if clip_tensor.shape[0] in (1, 3):
            pass
        elif clip_tensor.shape[1] in (1, 3):
            clip_tensor = clip_tensor.permute(1, 0, 2, 3)
        elif clip_tensor.shape[-1] in (1, 3):
            clip_tensor = clip_tensor.permute(-1, 0, 1, 2)
        else:
            raise ValueError(f"Cannot infer channel dimension for clip shape {clip_tensor.shape}")
        if clip_tensor.ndim == 4:
            clip_tensor = clip_tensor.unsqueeze(0)
    elif clip_tensor.ndim == 5:
        if clip_tensor.shape[1] not in (1, 3):
            channel_axis = next((i for i, s in enumerate(clip_tensor.shape) if s in (1, 3)), None)
            if channel_axis is None:
                raise ValueError(f"Cannot infer channel dimension for clip shape {clip_tensor.shape}")
            perm = [0, channel_axis] + [i for i in range(1, clip_tensor.ndim) if i != channel_axis]
            clip_tensor = clip_tensor.permute(*perm)
    else:
        raise ValueError(f"Unsupported clip shape {clip_tensor.shape}")

    return clip_tensor.contiguous()


class ViolenceDetectorModelB:
    """
    MoViNet-based streaming violence detector.
    
    This model:
    1. Collects frames into clips
    2. Runs MoViNet inference on clips
    3. Returns violence probability
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        clip_length: int = 6,
        threshold: float = 0.47,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize Model B.
        
        Args:
            checkpoint_path: Path to MoViNet checkpoint
            device: Device for inference ('cpu' or 'cuda')
            clip_length: Number of frames per clip
            threshold: Violence probability threshold
            class_names: List of class names (default: ["no_violence", "violence"])
        """
        self.device = device
        self.clip_length = clip_length
        self.threshold = threshold
        self.class_names = class_names or ["no_violence", "violence"]
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=clip_length)
        self.last_prob = 0.0
        self.last_inference_time = 0.0
        
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load MoViNet model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Build model
        num_classes = len(self.class_names)
        model = build_movinet_a0_stream(num_classes, pretrained=False)
        model.to(self.device)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract state dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        
        # Handle DataParallel keys
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        
        model.load_state_dict(state, strict=True)
        model.eval()
        
        # Clear buffers for streaming
        if hasattr(model, "clean_activation_buffers"):
            model.clean_activation_buffers()
        
        return model
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (annotated_frame, violence_probability, inference_ran)
        """
        # Preprocess and add to buffer
        processed = preprocess_frame(frame)
        self.frame_buffer.append(processed.clone())
        
        inference_ran = False
        
        # Run inference when buffer is full
        if len(self.frame_buffer) >= self.clip_length:
            clip = list(self.frame_buffer)
            self.last_prob = self._run_inference(clip)
            inference_ran = True
            self.last_inference_time = time.time()
            
            # Clear buffer after inference
            self.frame_buffer.clear()
            
            # Clear model buffers for next clip
            if hasattr(self.model, "clean_activation_buffers"):
                self.model.clean_activation_buffers()
        
        # Annotate frame
        annotated = self._annotate_frame(frame.copy(), self.last_prob)
        
        return annotated, self.last_prob, inference_ran
    
    def _run_inference(self, clip: List[torch.Tensor]) -> float:
        """Run inference on a clip."""
        clip_tensor = prepare_clip_tensor(clip).to(self.device, non_blocking=True)
        
        with torch.no_grad():
            logits = self.model(clip_tensor)
            probs = F.softmax(logits, dim=1)
            prob_vec = probs.squeeze(0).cpu().numpy()
        
        # Return violence probability (assume class 1 is violence)
        violence_idx = 1 if len(prob_vec) > 1 else 0
        return float(prob_vec[violence_idx])
    
    def _annotate_frame(self, frame: np.ndarray, prob: float) -> np.ndarray:
        """Draw violence probability on frame."""
        color = (0, 0, 255) if prob >= self.threshold else (0, 255, 0)
        label = "VIOLENCE" if prob >= self.threshold else "NO VIOLENCE"
        text = f"{label}: {prob:.2f}"
        
        cv2.putText(
            frame, 
            text, 
            (12, 32), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            color, 
            2
        )
        
        # Timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            ts,
            (12, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        
        return frame
    
    def reset(self):
        """Reset detector state."""
        self.frame_buffer.clear()
        self.last_prob = 0.0
        self.last_inference_time = 0.0
        
        if hasattr(self.model, "clean_activation_buffers"):
            self.model.clean_activation_buffers()
