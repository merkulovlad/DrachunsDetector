# FastAPI Violence Detection Service

Concise guide for running and using the FastAPI wrapper around the violence detection models.

## Overview
- Two models: ST-GCN (skeleton-based) and MoViNet (video-based).
- Two modes: live monitoring (`/live`) and offline analysis (`/offline`).
- Built with FastAPI, WebSockets, MJPEG streaming, and background tasks.

## Quick start
1) Ensure Python 3.10+. From `app/`, install dependencies:
```
## 🎯 Features

### Multiple Detection Models
- **Model A (ST-GCN)**: Skeleton-based violence detection using Spatial-Temporal Graph Convolutional Networks with YOLO pose estimation
- **Model B (MoViNet)**: Video-based violence detection using Mobile Video Networks (streaming variant)
- **Model C (R2P1 / R(2+1)D-18)**: Clip-based ResNet-style video encoder trained on violence clips
- **Model D (ViViT)**: Transformer-based video classifier loaded from Hugging Face weights
- **Model E (VideoMAE)**: Transformer-based masked autoencoder fine-tuned for violence detection

### Two Analysis Modes
1. **Live Monitor** (`/live`): Real-time violence detection from webcam or phone camera
   - Local webcam support
   - Browser-based video capture (phone cameras)
   - WebSocket streaming with annotated frames
   - MJPEG stream support
   - Real-time model switching

2. **Offline Analyzer** (`/offline`): Upload and analyze pre-recorded videos
   - Drag-and-drop video upload
   - Background processing with progress tracking
   - Violence detection timeline
   - Downloadable annotated videos

## 🏗️ Architecture

```
app/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── api/                    # API routes
│   ├── live.py            # Live streaming endpoints
│   └── offline.py         # Offline processing endpoints
├── models/                 # Model wrapper classes
│   ├── violence_a.py      # Model A (ST-GCN) wrapper
│   ├── violence_b.py      # Model B (MoViNet) wrapper
│   ├── violence_r2p1.py   # Model C (R2P1) wrapper
│   ├── violence_transformers.py  # Model D/E (ViViT/VideoMAE) wrappers
│   └── video_utils.py     # Shared video helpers
├── core/                   # Core utilities
│   ├── config.py          # Configuration settings
│   └── model_manager.py   # Model instance management
├── templates/              # HTML templates
│   ├── live.html          # Live monitor page
│   └── offline.html       # Offline analyzer page
├── static/                 # Static files (CSS, JS, images)
├── uploads/                # Temporary uploaded videos
└── outputs/                # Processed output videos
```

## 📋 Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for faster processing)
- Webcam or mobile device with camera (for live monitoring)
- Pre-trained model checkpoints:
  - Model A: ST-GCN checkpoint file (`.pt`)
  - Model B: MoViNet checkpoint file (`.pt`)
  - Model C: R2P1 checkpoint file (`.pt`)
  - Model D: ViViT model directory (config + safetensors)
  - Model E: VideoMAE model directory (config + safetensors)

## 🚀 Installation

### 1. Clone the Repository

```bash
cd /Users/vladislav/DrachunsDetector
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
cd app
pip install -r requirements.txt
```
2) Create your env file and set checkpoint paths:
```
cp .env.example .env
# edit .env to point to model weights and data directories
```
3) Launch the service:
```
./start.sh
```
4) Open the UI at `http://localhost:8000/live` (live) or `http://localhost:8000/offline` (uploads). API docs: `http://localhost:8000/docs`.

**Note**: This installs all detector dependencies (ST-GCN, MoViNet, R2P1, ViViT, and VideoMAE). The MoViNet library is installed directly from GitHub alongside Hugging Face's `transformers`.

### 4. Download or Place Model Checkpoints

Place your trained model checkpoints in accessible locations:

```bash
# Example locations (adjust as needed)
# Model A checkpoint: /path/to/stgcn_checkpoint.pt
# Model B checkpoint: /path/to/movinet_checkpoint.pt
```

### 5. Configure Environment Variables (Optional)

Create a `.env` file in the `app/` directory:

```env
# Model A settings
MODEL_A_CHECKPOINT=/path/to/stgcn_checkpoint.pt
MODEL_A_DEVICE=cpu  # or cuda
MODEL_A_THRESHOLD=0.8

# Model B settings
MODEL_B_CHECKPOINT=/path/to/movinet_checkpoint.pt
MODEL_B_DEVICE=cpu  # or cuda
MODEL_B_THRESHOLD=0.47

# Model C (R2P1) settings
MODEL_C_CHECKPOINT=./r2p1_vivit_mae/r2p1/best_r2p1d18.pt
MODEL_C_DEVICE=cpu
MODEL_C_CLIP_LENGTH=16
MODEL_C_THRESHOLD=0.5

# Model D (ViViT) settings
MODEL_D_CHECKPOINT=./r2p1_vivit_mae/vivit/best_model
MODEL_D_DEVICE=cpu
MODEL_D_CLIP_LENGTH=32
MODEL_D_THRESHOLD=0.5
MODEL_D_POSITIVE_LABEL=Fight

# Model E (VideoMAE) settings
MODEL_E_CHECKPOINT=./r2p1_vivit_mae/mae/best_model
MODEL_E_DEVICE=cpu
MODEL_E_CLIP_LENGTH=16
MODEL_E_THRESHOLD=0.5
MODEL_E_POSITIVE_LABEL=Fight

# Server settings
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

Alternatively, you can set these when starting the server or modify `core/config.py` directly.

## 🎮 Usage

### Starting the Server

#### Option 1: With Environment Variables

```bash
cd app
python main.py
```

#### Option 2: Programmatic Initialization

```python
from app.main import app
from app.core.model_manager import model_manager
import uvicorn

# Initialize models with checkpoint paths
model_manager.initialize_models(
    model_a_checkpoint="/path/to/stgcn_checkpoint.pt",
    model_b_checkpoint="/path/to/movinet_checkpoint.pt"
)

# Run server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Option 3: Direct Uvicorn

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the Application

Once the server is running, open your browser and navigate to:

- **Home**: http://localhost:8000
- **Live Monitor**: http://localhost:8000/live
- **Offline Analyzer**: http://localhost:8000/offline
- **API Documentation**: http://localhost:8000/docs

## 📱 Using Live Monitor

### Method 1: Local Webcam

1. Go to http://localhost:8000/live
2. Select "Camera Stream" tab
3. Choose your model (A or B)
4. Click "Start Camera"
5. The live feed will appear with real-time annotations

### Method 2: Phone/Browser Capture

1. Go to http://localhost:8000/live
2. Select "Browser Capture" tab
3. Choose your model
4. Click "Start Capture"
5. Grant camera permissions when prompted
6. The feed will be sent to the server and processed frames returned

### WebSocket API

For custom integrations:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/live/stream?model=a');
ws.binaryType = 'arraybuffer';

ws.onopen = () => {
    // Send JPEG-encoded frames
    canvas.toBlob(blob => {
        blob.arrayBuffer().then(buffer => {
            ws.send(buffer);
        });
    }, 'image/jpeg');
};

ws.onmessage = (event) => {
    // Receive annotated frames
    const blob = new Blob([event.data], {type: 'image/jpeg'});
    const url = URL.createObjectURL(blob);
    img.src = url;
};
```

## 📹 Using Offline Analyzer

1. Go to http://localhost:8000/offline
2. Select your model (A or B)
3. Upload a video file:
   - Click "Choose Video File", or
   - Drag and drop a video file
4. Click "Start Analysis"
5. Monitor progress in real-time
6. View results:
   - Violence detection timeline
   - Frame-by-frame analysis
   - Download processed video

### API Usage

```python
import requests

# Upload video
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/offline/upload?model=a',
        files={'file': f}
    )
    job_id = response.json()['job_id']

# Poll status
while True:
    status = requests.get(f'http://localhost:8000/api/offline/status/{job_id}').json()
    print(f"Progress: {status['progress']}%")
    if status['status'] == 'completed':
        break
    time.sleep(1)

# Get results
results = requests.get(f'http://localhost:8000/api/offline/result/{job_id}').json()
print(f"Violence detected: {results['violence_detected']}")

# Download processed video
video = requests.get(f'http://localhost:8000/api/offline/download/{job_id}')
with open('processed.mp4', 'wb') as f:
    f.write(video.content)
```

## Key endpoints
- Live monitoring: `/live`, `/api/live/stream` (WebSocket), `/api/live/camera/stream` (MJPEG), `/api/live/status`.
- Offline processing: `/offline`, `/api/offline/upload`, `/api/offline/status/{job_id}`, `/api/offline/result/{job_id}`, `/api/offline/download/{job_id}`.
- Model control: `/api/model/select`, `/api/model/current`.

## Configuration
- `.env` controls model paths, device selection, and server settings.
- `core/config.py` centralizes defaults.
- `start.sh` applies sensible defaults and starts Uvicorn.

## Project structure (app/)
```
app/
├── main.py              # FastAPI entrypoint
├── api/                 # Live and offline route handlers
├── models/              # Wrappers: ST-GCN (violence_a), MoViNet (violence_b)
├── core/                # Settings and model manager
├── templates/           # live.html, offline.html
├── static/              # Assets placeholder
├── uploads/             # Temp uploads
├── outputs/             # Processed outputs
├── start.sh             # Launch script
└── run_example.py       # Minimal init example
```

## Notes
- Requires model checkpoints for both models; set paths in `.env`.
- Background tasks are used for offline jobs; progress is reported via `/api/offline/status/{job_id}`.
- For production, run behind a process manager (e.g., systemd, supervisord) and place a reverse proxy (e.g., Nginx) in front of Uvicorn.
