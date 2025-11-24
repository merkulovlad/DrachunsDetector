"""
Example script showing how to initialize and run the Violence Detection System.
"""
import sys
from pathlib import Path

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from main import app
from core.model_manager import model_manager
import uvicorn

# Configuration
MODEL_A_CHECKPOINT = "/path/to/your/stgcn_checkpoint.pt"  # Update this
MODEL_B_CHECKPOINT = "/path/to/your/movinet_checkpoint.pt"  # Update this

def main():
    """Initialize models and start the server."""
    
    print("=" * 60)
    print("🚀 Violence Detection System")
    print("=" * 60)
    print()
    
    # Check if checkpoint paths are valid
    model_a_path = Path(MODEL_A_CHECKPOINT)
    model_b_path = Path(MODEL_B_CHECKPOINT)
    
    print("📍 Checkpoint locations:")
    print(f"   Model A: {MODEL_A_CHECKPOINT}")
    print(f"   Model B: {MODEL_B_CHECKPOINT}")
    print()
    
    # Initialize models
    print("🔧 Initializing models...")
    try:
        model_manager.initialize_models(
            model_a_checkpoint=MODEL_A_CHECKPOINT if model_a_path.exists() else None,
            model_b_checkpoint=MODEL_B_CHECKPOINT if model_b_path.exists() else None,
        )
        
        available_models = model_manager.list_available_models()
        print(f"✅ Available models: {available_models}")
        
        if not available_models:
            print("⚠️  Warning: No models initialized!")
            print("   Please update the checkpoint paths in this script.")
            print()
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        print()
    
    print("=" * 60)
    print("🌐 Starting server...")
    print()
    print("   🔗 URLs:")
    print("      - Home:             http://localhost:8000")
    print("      - Live Monitor:     http://localhost:8000/live")
    print("      - Offline Analyzer: http://localhost:8000/offline")
    print("      - API Docs:         http://localhost:8000/docs")
    print()
    print("   Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
