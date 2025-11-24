"""
Transformer-based video classifiers (ViViT / VideoMAE) for violence detection.
"""
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification


class HuggingFaceVideoClassifier:
    """Generic buffered video classifier built on top of Hugging Face models."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        clip_length: int = 16,
        threshold: float = 0.5,
        positive_label: str = "Fight",
    ):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.clip_length = clip_length
        self.threshold = threshold
        self.positive_label_name = positive_label

        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self.model = AutoModelForVideoClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.id2label: Dict[int, str] = {}
        raw_mapping = getattr(self.model.config, "id2label", None)
        if raw_mapping:
            self.id2label = {int(k): v for k, v in raw_mapping.items()}

        self.positive_index = self._infer_positive_index(positive_label)
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=clip_length)
        self.last_prob: float = 0.0

    def _infer_positive_index(self, label_name: str) -> int:
        if self.id2label:
            for idx, name in self.id2label.items():
                if name.lower() == label_name.lower():
                    return idx
            for idx, name in self.id2label.items():
                if label_name.lower() in name.lower():
                    return idx
            return max(self.id2label.keys())
        return 1

    def process_frame(self, frame: np.ndarray):
        """Add frame to buffer, run inference when clip is ready, and annotate."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_buffer.append(rgb)

        inference_ran = False
        if len(self.frame_buffer) >= self.clip_length:
            clip = list(self.frame_buffer)
            self.last_prob = self._run_inference(clip)
            self.frame_buffer.clear()
            inference_ran = True

        annotated = self._annotate_frame(frame.copy(), self.last_prob)
        return annotated, self.last_prob, inference_ran

    def _run_inference(self, clip: List[np.ndarray]) -> float:
        # Ensure consistent numpy arrays per frame; keep as list for HF processor compatibility
        clip_array = [np.asarray(f, dtype=np.uint8) for f in clip]
        inputs = self.processor(clip_array, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=-1)

        idx = self.positive_index if probs.shape[-1] > self.positive_index else probs.shape[-1] - 1
        return float(probs.squeeze(0)[idx].item())

    def _annotate_frame(self, frame: np.ndarray, prob: float) -> np.ndarray:
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


class ViolenceDetectorViViT(HuggingFaceVideoClassifier):
    """ViViT-based classifier wrapper."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ViolenceDetectorVideoMAE(HuggingFaceVideoClassifier):
    """VideoMAE-based classifier wrapper."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
