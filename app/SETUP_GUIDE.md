# 🚀 FastAPI Violence Detection System - Setup Guide

Welcome! This guide will help you get the web application up and running.

## ✅ Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.10 or higher installed
- [ ] Git (to clone dependencies)
- [ ] At least 4GB RAM available
- [ ] Model checkpoint files:
  - [ ] Model A checkpoint (ST-GCN `.pt` file)
  - [ ] Model B checkpoint (MoViNet `.pt` file)
- [ ] (Optional) CUDA-capable GPU for faster processing

## 📦 Step-by-Step Installation

### Step 1: Navigate to the App Directory

```bash
cd /Users/vladislav/DrachunsDetector/app
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or on Windows:
# venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This will install:
- FastAPI and Uvicorn (web server)
- PyTorch and torchvision (deep learning)
- Ultralytics YOLO (pose estimation)
- MoViNet (video networks)
- OpenCV (video processing)
- And all other dependencies

**Note**: Installation may take 5-10 minutes depending on your internet speed.

### Step 4: Configure Model Checkpoints

You have three options:

#### Option A: Environment File (Recommended)

```bash
# Copy the example
cp .env.example .env

# Edit with your favorite editor
nano .env  # or vim, code, etc.
```

Update these lines with your checkpoint paths:

```env
MODEL_A_CHECKPOINT=/path/to/your/stgcn_checkpoint.pt
MODEL_B_CHECKPOINT=/path/to/your/movinet_checkpoint.pt
```

#### Option B: Edit run_example.py

```bash
# Open the file
nano run_example.py
```

Update these lines:

```python
MODEL_A_CHECKPOINT = "/path/to/your/stgcn_checkpoint.pt"
MODEL_B_CHECKPOINT = "/path/to/your/movinet_checkpoint.pt"
```

#### Option C: Set Environment Variables

```bash
export MODEL_A_CHECKPOINT="/path/to/your/stgcn_checkpoint.pt"
export MODEL_B_CHECKPOINT="/path/to/your/movinet_checkpoint.pt"
```

### Step 5: Start the Server

Choose one method:

#### Method 1: Quick Start Script

```bash
./start.sh
```

#### Method 2: Run Example Script

```bash
python run_example.py
```

#### Method 3: Direct Python

```bash
python main.py
```

#### Method 4: Uvicorn (Development)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see output like:

```
🚀 Violence Detection System
============================================================

📍 Checkpoint locations:
   Model A: /path/to/stgcn.pt
   Model B: /path/to/movinet.pt

🔧 Initializing models...
✓ Model A initialized from /path/to/stgcn.pt
✓ Model B initialized from /path/to/movinet.pt
✅ Available models: ['a', 'b']

============================================================
🌐 Starting server...

   🔗 URLs:
      - Home:             http://localhost:8000
      - Live Monitor:     http://localhost:8000/live
      - Offline Analyzer: http://localhost:8000/offline
      - API Docs:         http://localhost:8000/docs

   Press Ctrl+C to stop
============================================================

INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Access the Application

Open your browser and go to:

1. **Home Page**: http://localhost:8000
   - Overview and navigation

2. **Live Monitor**: http://localhost:8000/live
   - Real-time camera streaming
   - Phone camera capture

3. **Offline Analyzer**: http://localhost:8000/offline
   - Upload and process videos

4. **API Documentation**: http://localhost:8000/docs
   - Interactive API documentation

## 🎯 Quick Test

### Test Live Monitor

1. Go to http://localhost:8000/live
2. Select "Camera Stream"
3. Choose Model A or B
4. Click "Start Camera"
5. You should see your webcam feed with annotations

### Test Offline Analyzer

1. Go to http://localhost:8000/offline
2. Choose a model
3. Upload a video file (any format: mp4, avi, mov, etc.)
4. Click "Start Analysis"
5. Watch the progress bar
6. View results and download processed video

## 🐛 Common Issues and Solutions

### Issue: "Model A not initialized"

**Problem**: Checkpoint file not found

**Solution**:
```bash
# Check if file exists
ls -lh /path/to/your/stgcn_checkpoint.pt

# Verify path in .env or run_example.py
cat .env | grep MODEL_A

# Make sure path is absolute, not relative
```

### Issue: "Import could not be resolved"

**Problem**: Dependencies not installed or wrong Python environment

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: "Camera not working"

**Problem**: Camera permissions or already in use

**Solution**:
```bash
# Check if camera is available
# Try different camera index: 0, 1, 2

# Close other apps using camera (Zoom, Skype, etc.)

# On macOS: System Preferences → Security & Privacy → Camera
# Grant terminal/Python camera access
```

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"

**Problem**: Dependencies not installed

**Solution**:
```bash
# Activate venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# If still failing, install individually
pip install ultralytics
pip install fastapi uvicorn
```

### Issue: "CUDA out of memory"

**Problem**: GPU memory insufficient

**Solution**:
```bash
# Use CPU instead
# In .env:
MODEL_A_DEVICE=cpu
MODEL_B_DEVICE=cpu

# Or reduce batch size/resolution
```

### Issue: Port 8000 already in use

**Problem**: Another service using port 8000

**Solution**:
```bash
# Use a different port
# In .env:
PORT=8080

# Or kill the process using port 8000
lsof -ti:8000 | xargs kill -9
```

## 📱 Using with Phone Camera

### Option 1: Browser Capture

1. Open http://localhost:8000/live on your phone
2. Select "Browser Capture"
3. Click "Start Capture"
4. Grant camera permissions
5. Video will stream to server and back

**Note**: Phone and computer must be on same network.

### Option 2: IP Webcam App (Android)

1. Install IP Webcam app
2. Start server in app
3. Note the RTSP URL
4. Use custom streaming solution or modify code

## 🔧 Advanced Configuration

### Custom Thresholds

Edit `app/core/config.py`:

```python
class Settings:
    model_a_threshold: float = 0.8  # Lower = more sensitive
    model_b_threshold: float = 0.47
```

### Custom Model Parameters

```python
# Model A (ST-GCN)
model_a_seq_len: int = 30      # Longer = more context
model_a_stride: int = 15       # Smaller = more frequent inference

# Model B (MoViNet)
model_b_clip_length: int = 6   # Frames per clip
```

### CORS Configuration

For production with different frontend:

```python
# In .env
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

## 📊 Performance Optimization

### Use GPU

```env
# In .env
MODEL_A_DEVICE=cuda
MODEL_B_DEVICE=cuda
```

### Reduce Resolution

Lower resolution = faster processing:

```python
# Modify in model wrappers or preprocessing
# For Model A: reduce YOLO image size
# For Model B: resize frames before preprocessing
```

### Choose Faster Model

Model B (MoViNet) is generally faster than Model A for most scenarios.

## 🔒 Security for Production

Before deploying to production:

1. **Disable Debug Mode**
   ```env
   DEBUG=false
   ```

2. **Configure CORS**
   ```env
   CORS_ORIGINS=https://yourdomain.com
   ```

3. **Add Authentication**
   ```python
   # Add middleware in main.py
   from fastapi import Depends, HTTPException
   from fastapi.security import HTTPBearer
   ```

4. **Use HTTPS**
   ```bash
   # Set up SSL certificate
   uvicorn main:app --ssl-keyfile key.pem --ssl-certfile cert.pem
   ```

5. **Set Up Reverse Proxy**
   ```nginx
   # nginx config
   server {
       listen 80;
       server_name yourdomain.com;
       
       location / {
           proxy_pass http://127.0.0.1:8000;
       }
   }
   ```

## 📚 Next Steps

1. ✅ Server is running
2. 📖 Read the full [README.md](README.md) for detailed documentation
3. 🧪 Test both models on your videos
4. ⚙️ Adjust thresholds based on your use case
5. 🚀 Deploy to production (if needed)

## 💬 Getting Help

If you encounter issues:

1. Check the error messages in terminal
2. Review this setup guide
3. Check the troubleshooting section in [README.md](README.md)
4. Review API docs at http://localhost:8000/docs
5. Check if all dependencies are installed: `pip list`

## 🎉 You're All Set!

Your Violence Detection System is now running!

- **Live Monitor**: http://localhost:8000/live
- **Offline Analyzer**: http://localhost:8000/offline
- **API Docs**: http://localhost:8000/docs

Enjoy using the system! 🚀

---

**Last Updated**: November 2024
