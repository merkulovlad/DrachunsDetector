"""Model manager for handling model selection and instances."""
from typing import Optional, Literal, Callable, Dict
from pathlib import Path
import sys

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ST-GCN"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "stream_platform"))

import torch
from models.violence_a import ViolenceDetectorModelA
from models.violence_b import ViolenceDetectorModelB
from models.violence_r2p1 import ViolenceDetectorModelR2P1
from models.violence_transformers import (
    ViolenceDetectorViViT,
    ViolenceDetectorVideoMAE,
)
from core.config import settings


ModelType = Literal["a", "b", "c", "d", "e"]
CHECKPOINT_FIELDS: Dict[ModelType, str] = {
    "a": "model_a_checkpoint",
    "b": "model_b_checkpoint",
    "c": "model_c_checkpoint",
    "d": "model_d_checkpoint",
    "e": "model_e_checkpoint",
}


class ModelManager:
    """Singleton manager for violence detection models."""
    
    _instance = None
    _models = {}
    _current_model: ModelType = "a"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize_models(
        self,
        model_a_checkpoint: Optional[str] = None,
        model_b_checkpoint: Optional[str] = None,
        model_c_checkpoint: Optional[str] = None,
        model_d_checkpoint: Optional[str] = None,
        model_e_checkpoint: Optional[str] = None,
    ):
        """
        Initialize the models with checkpoint paths.
        
        Args:
            model_a_checkpoint: Path to Model A (ST-GCN) checkpoint
            model_b_checkpoint: Path to Model B (MoViNet) checkpoint
            model_c_checkpoint: Path to Model C (R(2+1)D) checkpoint
            model_d_checkpoint: Path to Model D (ViViT) directory
            model_e_checkpoint: Path to Model E (VideoMAE) directory
        """
        # Update settings if paths provided
        if model_a_checkpoint:
            settings.model_a_checkpoint = model_a_checkpoint
        if model_b_checkpoint:
            settings.model_b_checkpoint = model_b_checkpoint
        if model_c_checkpoint:
            settings.model_c_checkpoint = model_c_checkpoint
        if model_d_checkpoint:
            settings.model_d_checkpoint = model_d_checkpoint
        if model_e_checkpoint:
            settings.model_e_checkpoint = model_e_checkpoint

        # Normalize device strings (treat "gpu" as "cuda" when available)
        def _resolve_device(value: str) -> str:
            requested = (value or "").lower()
            if requested in ("gpu", "cuda", "cuda:0"):
                return "cuda" if torch.cuda.is_available() else "cpu"
            if requested.startswith("cuda"):
                return "cuda" if torch.cuda.is_available() else "cpu"
            return requested or "cpu"
        
        device_a = _resolve_device(settings.model_a_device)
        device_b = _resolve_device(settings.model_b_device)
        device_c = _resolve_device(settings.model_c_device)
        device_d = _resolve_device(settings.model_d_device)
        device_e = _resolve_device(settings.model_e_device)
        
        # Initialize Model A
        if settings.model_a_checkpoint and Path(settings.model_a_checkpoint).exists():
            self._init_model(
                "a",
                settings.model_a_checkpoint,
                lambda: ViolenceDetectorModelA(
                    clf_checkpoint=settings.model_a_checkpoint,
                    pose_weights=settings.model_a_pose_weights,
                    device=device_a,
                    seq_len=settings.model_a_seq_len,
                    stride=settings.model_a_stride,
                    threshold=settings.model_a_threshold,
                ),
            )
        
        # Initialize Model B
        if settings.model_b_checkpoint and Path(settings.model_b_checkpoint).exists():
            self._init_model(
                "b",
                settings.model_b_checkpoint,
                lambda: ViolenceDetectorModelB(
                    checkpoint_path=settings.model_b_checkpoint,
                    device=device_b,
                    clip_length=settings.model_b_clip_length,
                    threshold=settings.model_b_threshold,
                    class_names=settings.model_b_class_names,
                ),
            )

        # Initialize Model C (R(2+1)D)
        if settings.model_c_checkpoint and Path(settings.model_c_checkpoint).exists():
            self._init_model(
                "c",
                settings.model_c_checkpoint,
                lambda: ViolenceDetectorModelR2P1(
                    checkpoint_path=settings.model_c_checkpoint,
                    device=device_c,
                    clip_length=settings.model_c_clip_length,
                    threshold=settings.model_c_threshold,
                ),
            )

        # Initialize Model D (ViViT)
        if settings.model_d_checkpoint and Path(settings.model_d_checkpoint).exists():
            self._init_model(
                "d",
                settings.model_d_checkpoint,
                lambda: ViolenceDetectorViViT(
                    model_dir=settings.model_d_checkpoint,
                    device=device_d,
                    clip_length=settings.model_d_clip_length,
                    threshold=settings.model_d_threshold,
                    positive_label=settings.model_d_positive_label,
                ),
            )

        # Initialize Model E (VideoMAE)
        if settings.model_e_checkpoint and Path(settings.model_e_checkpoint).exists():
            self._init_model(
                "e",
                settings.model_e_checkpoint,
                lambda: ViolenceDetectorVideoMAE(
                    model_dir=settings.model_e_checkpoint,
                    device=device_e,
                    clip_length=settings.model_e_clip_length,
                    threshold=settings.model_e_threshold,
                    positive_label=settings.model_e_positive_label,
                ),
            )
        
        # Auto-select first available model as current
        self.auto_select_model()

    def _init_model(self, key: ModelType, checkpoint: str, factory: Callable[[], object]):
        """Safely initialize a detector and store it."""
        try:
            self._models[key] = factory()
            print(f"[✓] Model {key.upper()} initialized from {checkpoint}")
        except Exception as exc:
            print(f"[✗] Failed to initialize Model {key.upper()}: {exc}")
    
    def get_model(self, model_type: Optional[ModelType] = None):
        """Get a model instance."""
        if model_type is None:
            model_type = self._current_model
        
        if model_type not in self._models:
            raise ValueError(f"Model {model_type} not initialized")
        
        return self._models[model_type]
    
    def set_current_model(self, model_type: ModelType):
        """Set the current active model."""
        if model_type not in self._models:
            raise ValueError(f"Model {model_type} not initialized")
        self._current_model = model_type
    
    def get_current_model_type(self) -> ModelType:
        """Get the current model type."""
        return self._current_model
    
    def auto_select_model(self):
        """Auto-select first available model as current."""
        if self._models:
            first_available = list(self._models.keys())[0]
            self._current_model = first_available
    
    def list_available_models(self) -> list:
        """List available initialized models."""
        return list(self._models.keys())
    
    def reset_model(self, model_type: Optional[ModelType] = None):
        """Reset a model's state."""
        if model_type is None:
            model_type = self._current_model
        
        if model_type in self._models:
            self._models[model_type].reset()


# Global instance
model_manager = ModelManager()
