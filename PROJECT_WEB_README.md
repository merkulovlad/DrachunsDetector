# Violence Detection System - FastAPI Web Application

This directory contains the new **FastAPI-based web application** for violence detection, providing both real-time and offline analysis capabilities with two different deep learning models.

## 🎯 Quick Start

### 1. Navigate to the app directory

```bash
cd app
```

### 2. Set up environment

```bash
# Option A: Use the start script (recommended)
./start.sh

# Option B: Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure model checkpoints

Copy and edit the environment file:

```bash
cp .env.example .env
# Edit .env and set your model checkpoint paths
```

### 4. Run the application

```bash
# Using the run example
python run_example.py

# Or directly
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open in browser

- **Live Monitor**: http://localhost:8000/live
- **Offline Analyzer**: http://localhost:8000/offline
- **API Docs**: http://localhost:8000/docs

## 📁 Project Structure

```
app/
├── README.md              # Comprehensive documentation
├── requirements.txt       # Python dependencies
├── .env.example          # Environment configuration template
├── start.sh              # Quick start script
├── run_example.py        # Example startup script
├── main.py               # FastAPI application
├── api/                  # API endpoints
│   ├── live.py          # Live streaming routes
│   └── offline.py       # Offline processing routes
├── models/               # Model wrappers
│   ├── violence_a.py    # ST-GCN model wrapper
│   └── violence_b.py    # MoViNet model wrapper
├── core/                 # Core functionality
│   ├── config.py        # Settings and configuration
│   └── model_manager.py # Model instance management
├── templates/            # HTML templates
│   ├── live.html        # Live monitor page
│   └── offline.html     # Offline analyzer page
├── static/              # Static assets (CSS, JS, images)
├── uploads/             # Temporary video uploads
└── outputs/             # Processed video outputs
```

## 🎨 Features

### Two Detection Models

1. **Model A**: ST-GCN (Skeleton-based)
   - YOLO pose estimation + graph convolutions
   - Privacy-preserving (skeleton data only)
   - Located in: `models/violence_a.py`

2. **Model B**: MoViNet (Video-based)
   - Streaming mobile video network
   - Holistic scene understanding
   - Located in: `models/violence_b.py`

### Two Analysis Modes

1. **Live Monitor** (`/live`)
   - Real-time webcam or phone camera
   - WebSocket streaming
   - MJPEG stream support
   - Switch models on-the-fly

2. **Offline Analyzer** (`/offline`)
   - Upload and process videos
   - Background job processing
   - Progress tracking
   - Violence timeline
   - Download processed videos

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Model A (ST-GCN)
MODEL_A_CHECKPOINT=/path/to/stgcn_checkpoint.pt
MODEL_A_DEVICE=cpu  # or cuda

# Model B (MoViNet)
MODEL_B_CHECKPOINT=/path/to/movinet_checkpoint.pt
MODEL_B_DEVICE=cpu  # or cuda

# Server
PORT=8000
DEBUG=false
```

### Programmatic Configuration

Edit `run_example.py` or create your own startup script:

```python
from app.core.model_manager import model_manager
import uvicorn

# Set checkpoint paths
model_manager.initialize_models(
    model_a_checkpoint="/path/to/stgcn.pt",
    model_b_checkpoint="/path/to/movinet.pt"
)

# Run server
uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
```

## 📊 API Endpoints

### Live Streaming

- `GET /live` - Live monitor page
- `WS /api/live/stream?model=a` - WebSocket stream
- `GET /api/live/camera/stream?model=a` - MJPEG stream
- `GET /api/live/camera/start` - Start camera
- `GET /api/live/camera/stop` - Stop camera

### Offline Processing

- `GET /offline` - Offline analyzer page
- `POST /api/offline/upload?model=a` - Upload video
- `GET /api/offline/status/{job_id}` - Check status
- `GET /api/offline/result/{job_id}` - Get results
- `GET /api/offline/download/{job_id}` - Download video

### Model Management

- `POST /api/model/select` - Select active model
- `GET /api/model/current` - Get current model

Full API documentation available at: http://localhost:8000/docs

## 🐛 Troubleshooting

### Import Errors

The model wrappers import from `main_code/` and `stream_platform/`. Ensure:

1. You're in the project root directory structure
2. Python path includes parent directories
3. Original model code is present

### Model Not Loading

Check:
- Checkpoint file paths are correct
- Files exist and are readable
- Dependencies are installed
- Check console output for errors

### Camera Issues

- Grant camera permissions in browser
- Try different camera index (0, 1, 2)
- Check if another app is using camera
- For phone: use HTTPS or localhost

## 📚 Documentation

See `README.md` in this directory for comprehensive documentation including:

- Detailed installation instructions
- Usage examples
- API reference
- Model details
- Performance notes
- Security considerations
- Development guide

## 🚀 Deployment

### Production Checklist

- [ ] Set `DEBUG=false` in `.env`
- [ ] Configure proper CORS origins
- [ ] Use HTTPS (SSL/TLS)
- [ ] Add authentication middleware
- [ ] Set up file cleanup jobs
- [ ] Configure reverse proxy (nginx)
- [ ] Use proper process manager (systemd, supervisor)
- [ ] Set up logging and monitoring

### Docker Deployment (Optional)

A `Dockerfile` can be created:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Integration with Existing Code

This web application wraps the existing models from:
- `../main_code/` - Model A (ST-GCN)
- `../stream_platform/` - Model B (MoViNet)

The original code remains unchanged. The wrappers in `models/` provide a clean interface for the web application.

## 📝 Next Steps

1. **Set Model Checkpoints**: Update paths in `.env` or `run_example.py`
2. **Test Locally**: Run the app and test both live and offline modes
3. **Customize**: Modify templates, add features, adjust thresholds
4. **Deploy**: Follow production checklist for deployment

## 💡 Tips

- Use Model B for faster processing
- Use Model A for privacy-sensitive scenarios
- Adjust thresholds based on your use case
- Monitor GPU memory usage
- Clean up old jobs periodically

---

**Need help?** Check the full README.md or API docs at `/docs`
