# FastAPI Violence Detection Service

Concise guide for running and using the FastAPI wrapper around the violence detection models.

## Overview
- Two models: ST-GCN (skeleton-based) and MoViNet (video-based).
- Two modes: live monitoring (`/live`) and offline analysis (`/offline`).
- Built with FastAPI, WebSockets, MJPEG streaming, and background tasks.

## Quick start
1) Ensure Python 3.10+. From `app/`, install dependencies:
```
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
