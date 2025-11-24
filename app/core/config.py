"""Configuration for the FastAPI application."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=('settings_',)
    )
    
    # App info
    app_name: str = "Violence Detection System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    uploads_dir: Path = base_dir / "uploads"
    outputs_dir: Path = base_dir / "outputs"
    
    # Model A settings (ST-GCN)
    model_a_checkpoint: str = ""  # Set via environment or at runtime
    model_a_pose_weights: str = "yolov8n-pose.pt"
    model_a_device: str = "cpu"
    model_a_seq_len: int = 30
    model_a_stride: int = 15
    model_a_threshold: float = 0.8
    
    # Model B settings (MoViNet)
    model_b_checkpoint: str = ""  # Set via environment or at runtime
    model_b_device: str = "cpu"
    model_b_clip_length: int = 6
    model_b_threshold: float = 0.47
    model_b_class_names: List[str] = ["no_violence", "violence"]
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS (can be comma-separated string from env, or list)
    cors_origins: str = "*"
    
    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from string or return as list."""
        if isinstance(self.cors_origins, str):
            if self.cors_origins == "*":
                return ["*"]
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins


settings = Settings()

# Ensure directories exist
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
