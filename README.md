# Violence Detection System

## Objective  
This project focuses on building a system that can **detect violent activity in video**, preferably in **real-time**.  

## Why It Matters  
By automatically identifying violent behavior, the system aims to:  
- Increase **public safety**  
- Enable **faster response times** for law enforcement


## Intended Users  
- Ministry of Internal Affairs  
- Police departments  
- Emergency services  

### How to setup 
- **Prerequisites**
  - Install Docker Desktop (macOS) or Docker Engine (Linux).
  - Install Python 3.10+ and create/activate your virtual environment (e.g., `conda activate <env>` or `python -m venv .venv && source .venv/bin/activate`).
  - Install FFmpeg (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Ubuntu/Debian).
  - Install project dependencies with `pip install -r requirements.txt`.

- **Start the MediaMTX relay**
  ```bash
  docker rm -f mediamtx 2>/dev/null || true
  docker run --name mediamtx --rm -it \
    -e MTX_WEBRTCALLOWORIGIN='*' \
    -e MTX_WEBRTCADDITIONALHOSTS='127.0.0.1,host.docker.internal,192.168.0.106' \
    -e MTX_RTSPTRANSPORTS=tcp \
    -p 8554:8554 \
    -p 1935:1935 \
    -p 8888:8888 \
    -p 8889:8889 \
    -p 8189:8189/udp \
    bluenviron/mediamtx:1
  ```
  This runs a temporary MediaMTX container that receives the RTSP stream and exposes the necessary ports.

- **Publish a local camera feed with FFmpeg**
  - macOS:
    ```bash
    ffmpeg -f avfoundation -framerate 30 -video_size 640x480 -i "0" \
      -c:v libx264 -vf "scale=640:480" -f rtsp rtsp://localhost:8554/mystream3
    ```
    If you have multiple cameras, list them with `ffmpeg -f avfoundation -list_devices true -i ""` and adjust the device index (the `"0"` above).
  - Linux:
    ```bash
    ffmpeg -f v4l2 -framerate 30 -video_size 640x480 -i /dev/video0 \
      -c:v libx264 -vf "scale=640:480" -f rtsp rtsp://localhost:8554/mystream3
    ```
    Replace `/dev/video0` with the correct video capture device if needed.

- **Run the backend**
  ```bash
  python -m stream_platform.backend.src.main
  ```
  The service connects to the RTSP stream (`rtsp://localhost:8554/mystream3`), performs inference, and reports detections in the console.

- **Verify everything is working**
  - Confirm the MediaMTX container logs show the RTSP client connection.
  - Watch the FFmpeg output to ensure frames are being published without errors.
  - In the Python process, look for log lines that mention frames being processed and detection results. If you enabled tracing or saved outputs in the configuration, check those locations as well.
