# FastAPI Violence Detection System

A comprehensive web-based violence detection system with two different deep learning models, offering both real-time streaming and offline video analysis capabilities.

## 🎯 Features

### Two Detection Models
- **Model A (ST-GCN)**: Skeleton-based violence detection using Spatial-Temporal Graph Convolutional Networks with YOLO pose estimation
- **Model B (MoViNet)**: Video-based violence detection using Mobile Video Networks (streaming variant)

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
│   └── violence_b.py      # Model B (MoViNet) wrapper
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

**Note**: This will install both Model A and Model B dependencies. The MoViNet library will be installed from GitHub.

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

## 🔧 API Endpoints

### Live Streaming

- `GET /live` - Live monitor page
- `GET /api/live/camera/start` - Start local camera
- `GET /api/live/camera/stop` - Stop camera
- `GET /api/live/camera/stream` - MJPEG stream endpoint
- `WS /api/live/stream` - WebSocket for bidirectional streaming
- `GET /api/live/status` - Get stream status

### Offline Processing

- `GET /offline` - Offline analyzer page
- `POST /api/offline/upload` - Upload video for processing
- `GET /api/offline/status/{job_id}` - Get job status
- `GET /api/offline/result/{job_id}` - Get job results
- `GET /api/offline/download/{job_id}` - Download processed video
- `GET /api/offline/jobs` - List all jobs
- `DELETE /api/offline/job/{job_id}` - Delete job

### Model Management

- `POST /api/model/select` - Select active model
- `GET /api/model/current` - Get current model

## 🎨 Model Details

### Model A: ST-GCN (Skeleton-Based)

**How it works:**
1. YOLO pose estimation detects persons and extracts 17 keypoints
2. Multi-object tracking assigns IDs to persons across frames
3. Skeleton sequences are built from tracked keypoints
4. ST-GCN processes skeleton graph sequences for violence classification

**Advantages:**
- Privacy-preserving (works on skeleton data)
- Robust to appearance changes
- Focuses on motion patterns

**Configuration:**
- Sequence length: 30 frames
- Stride: 15 frames
- Default threshold: 0.8

### Model B: MoViNet (Video-Based)

**How it works:**
1. Frames are preprocessed and collected into clips
2. MoViNet processes temporal video clips
3. Streaming architecture maintains temporal context
4. Outputs violence probability per clip

**Advantages:**
- Holistic scene understanding
- Efficient streaming architecture
- Pre-trained on large video datasets

**Configuration:**
- Clip length: 6 frames
- Default threshold: 0.47

## 🛠️ Configuration Options

### `app/core/config.py`

```python
class Settings:
    # Model A (ST-GCN)
    model_a_checkpoint: str
    model_a_pose_weights: str = "yolov8n-pose.pt"
    model_a_device: str = "cpu"
    model_a_seq_len: int = 30
    model_a_stride: int = 15
    model_a_threshold: float = 0.8
    
    # Model B (MoViNet)
    model_b_checkpoint: str
    model_b_device: str = "cpu"
    model_b_clip_length: int = 6
    model_b_threshold: float = 0.47
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
```

## 🐛 Troubleshooting

### Models Not Loading

**Issue**: "Model X not initialized"

**Solutions**:
- Verify checkpoint paths are correct
- Ensure checkpoint files exist and are accessible
- Check file permissions
- Review logs for specific error messages

### Camera Not Starting

**Issue**: Camera stream not working

**Solutions**:
- Check if another application is using the camera
- Verify camera permissions (browser/system)
- Try a different camera index (0, 1, 2...)
- Check browser console for errors

### WebSocket Connection Failed

**Issue**: Browser capture not working

**Solutions**:
- Ensure server is running on correct port
- Check firewall settings
- Use HTTPS for secure contexts (required on some browsers)
- Verify WebSocket URL format

### Slow Processing

**Issue**: Video processing is slow

**Solutions**:
- Use GPU: Set `device="cuda"` in config
- Reduce video resolution before upload
- Use Model B (generally faster than Model A)
- Adjust model parameters (seq_len, stride for Model A)

### Import Errors

**Issue**: Module not found errors

**Solutions**:
```bash
# Ensure paths are correctly added
export PYTHONPATH="${PYTHONPATH}:/Users/vladislav/DrachunsDetector"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📊 Performance Notes

### Model A (ST-GCN)
- **Speed**: ~15-30 FPS (CPU), ~60+ FPS (GPU)
- **Memory**: ~2-4 GB
- **Best for**: Crowded scenes, privacy-sensitive applications

### Model B (MoViNet)
- **Speed**: ~20-40 FPS (CPU), ~80+ FPS (GPU)
- **Memory**: ~3-6 GB
- **Best for**: General purpose, outdoor scenes

## 🔒 Security Considerations

1. **File Upload**: Validate file types and sizes
2. **CORS**: Configure appropriate origins in production
3. **Authentication**: Add authentication middleware for production
4. **HTTPS**: Use SSL/TLS for production deployments
5. **File Cleanup**: Implement automatic cleanup of old uploads/outputs

## 📝 Development

### Adding a New Model

1. Create wrapper class in `app/models/violence_c.py`
2. Implement required methods: `__init__`, `process_frame`, `reset`
3. Add configuration to `app/core/config.py`
4. Update `ModelManager` in `app/core/model_manager.py`
5. Update frontend dropdowns to include new model

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

## 🤝 Contributing

This is a project-specific implementation. For contributions:

1. Follow PEP 8 style guide
2. Add type hints
3. Update documentation
4. Test thoroughly before committing

## 📄 License

[Specify your license]

## 🙏 Acknowledgments

- **YOLO**: Ultralytics YOLOv8
- **ST-GCN**: Spatial Temporal Graph Convolutional Networks
- **MoViNet**: Mobile Video Networks
- **FastAPI**: Modern web framework
- **OpenCV**: Computer vision library

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Check logs for error messages

## 🔄 Version History

### v1.0.0 (Current)
- Initial release
- Two model support (ST-GCN, MoViNet)
- Live and offline processing
- WebSocket and MJPEG streaming
- Background job processing

---

**Built with ❤️ using FastAPI, PyTorch, and OpenCV**
