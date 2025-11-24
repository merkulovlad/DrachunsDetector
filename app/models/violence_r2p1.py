"""
Model C: R(2+1)D-based violence detector for clip-level inference.
"""
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision.models.video import r2plus1d_18

from models.video_utils import prepare_clip_tensor


class ViolenceDetectorModelR2P1:
    """Wrapper around an R(2+1)D-18 backbone fine-tuned for violence detection."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        clip_length: int = 16,
        threshold: float = 0.5,
    ):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.clip_length = clip_length
        self.threshold = threshold
        self.checkpoint_path = Path(checkpoint_path)
        self.frame_buffer: Deque[torch.Tensor] = deque(maxlen=clip_length)
        self.last_prob: float = 0.0
        self.model = self._load_model()
        self.frame_transform = T.Compose([
            T.ToImage(),
            T.ConvertImageDtype(torch.float32),
            T.Resize((128, 171)),
            T.CenterCrop((112, 112)),
            T.Normalize(
                mean=(0.43216, 0.394666, 0.37645),
                std=(0.22803, 0.22145, 0.216989),
            ),
        ])

    def _load_model(self) -> nn.Module:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"R(2+1)D checkpoint not found: {self.checkpoint_path}")

        model = r2plus1d_18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        state = ckpt
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]

        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Process a single BGR frame and return the annotated frame with probability."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.frame_transform(rgb)
        self.frame_buffer.append(tensor)

        inference_ran = False
        if len(self.frame_buffer) >= self.clip_length:
            clip = list(self.frame_buffer)
            self.last_prob = self._run_inference(clip)
            self.frame_buffer.clear()
            inference_ran = True

        annotated = self._annotate_frame(frame.copy(), self.last_prob)
        return annotated, self.last_prob, inference_ran

    def _run_inference(self, clip: List[torch.Tensor]) -> float:
        clip_tensor = prepare_clip_tensor(clip).to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.model(clip_tensor)
            probs = F.softmax(logits, dim=1)
        violence_idx = 1 if probs.shape[1] > 1 else 0
        return float(probs.squeeze(0)[violence_idx].item())

    def _annotate_frame(self, frame, prob: float):
        label = "VIOLENCE" if prob >= self.threshold else "NO VIOLENCE"
        color = (0, 0, 255) if prob >= self.threshold else (0, 255, 0)
        cv2.putText(
            frame,
            f"{label}: {prob:.2f}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )
        return frame

    def reset(self):
        self.frame_buffer.clear()
        self.last_prob = 0.0
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
