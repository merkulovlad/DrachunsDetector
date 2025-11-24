"""Model manager for handling model selection and instances."""
from typing import Optional, Literal
from pathlib import Path
import sys

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "main_code"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "stream_platform"))

from models.violence_a import ViolenceDetectorModelA
from models.violence_b import ViolenceDetectorModelB
from core.config import settings


ModelType = Literal["a", "b"]


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
    ):
        """
        Initialize the models with checkpoint paths.
        
        Args:
            model_a_checkpoint: Path to Model A (ST-GCN) checkpoint
            model_b_checkpoint: Path to Model B (MoViNet) checkpoint
        """
        # Update settings if paths provided
        if model_a_checkpoint:
            settings.model_a_checkpoint = model_a_checkpoint
        if model_b_checkpoint:
            settings.model_b_checkpoint = model_b_checkpoint
        
        # Initialize Model A if checkpoint available
        if settings.model_a_checkpoint and Path(settings.model_a_checkpoint).exists():
            try:
                self._models["a"] = ViolenceDetectorModelA(
                    clf_checkpoint=settings.model_a_checkpoint,
                    pose_weights=settings.model_a_pose_weights,
                    device=settings.model_a_device,
                    seq_len=settings.model_a_seq_len,
                    stride=settings.model_a_stride,
                    threshold=settings.model_a_threshold,
                )
                print(f"✓ Model A initialized from {settings.model_a_checkpoint}")
            except Exception as e:
                print(f"✗ Failed to initialize Model A: {e}")
        
        # Initialize Model B if checkpoint available
        if settings.model_b_checkpoint and Path(settings.model_b_checkpoint).exists():
            try:
                self._models["b"] = ViolenceDetectorModelB(
                    checkpoint_path=settings.model_b_checkpoint,
                    device=settings.model_b_device,
                    clip_length=settings.model_b_clip_length,
                    threshold=settings.model_b_threshold,
                    class_names=settings.model_b_class_names,
                )
                print(f"✓ Model B initialized from {settings.model_b_checkpoint}")
            except Exception as e:
                print(f"✗ Failed to initialize Model B: {e}")
        
        # Auto-select first available model as current
        self.auto_select_model()
    
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
