# Violence Detection System

Multi-model violence detection for live camera streams and offline video uploads. The FastAPI app wraps five detectors (ST-GCN, MoViNet, R(2+1)D, ViViT, VideoMAE) with a browser UI and REST/WebSocket APIs.

## Repository layout
- `app/` - FastAPI service (UI, APIs, model wrappers, checkpoints, configs).
- `ST-GCN/` - ST-GCN training/utilities and demos.
- `stream_platform/` - MoViNet streaming pipeline with MediaMTX helpers.
- `notebooks/` - Exploration and training notebooks.

## Prerequisites
- Git, Docker, and Docker Compose (for the containerized run).
- Or Python 3.10+ and pip (for local run).
- Optional GPU: NVIDIA driver + CUDA on the host; NVIDIA Container Toolkit if using Docker.

## Model checkpoints
The repo already contains example weights you can start with:
- ST-GCN: `app/checkpoints/best_model.pth`
- MoViNet: `app/checkpoints/epoch009.pt`
- R(2+1)D: `app/r2p1_vivit_mae/r2p1/best_r2p1d18.pt` (path set in `.env`)
- ViViT: `app/r2p1_vivit_mae/vivit/best_model/`
- VideoMAE: `app/r2p1_vivit_mae/mae/best_model/`

Update the `.env` values if you swap in your own checkpoints.

## Quick start (Docker Compose)
1) Clone and enter the repo:
```bash
git clone https://github.com/merkulovlad/DrachunsDetector.git
cd DrachunsDetector
```
2) Check or edit `.env` (at repo root). It feeds the container and already points to the bundled checkpoints; adjust device values to `cpu`/`cuda` as needed.
3) Build and run:
```bash
docker compose up --build
```
4) Open the app:
- Live monitor: http://localhost:8000/live
- Offline analyzer: http://localhost:8000/offline
- API docs: http://localhost:8000/docs

## Quick start (local Python)
1) Create a venv and install deps (from repo root):
```bash
cd app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2) Copy or edit `app/.env` to point to your checkpoints and choose devices.
3) Run the service:
```bash
./start.sh               # Windows: bash start.sh
# or: uvicorn main:app --host 0.0.0.0 --port 8000
```
4) Browse to the same URLs as above.

## Configuration (.env)
Key variables (both `./.env` for Docker and `app/.env` for local runs):
- `MODEL_A_*` - ST-GCN (checkpoint, pose weights, device, seq length/stride, threshold).
- `MODEL_B_*` - MoViNet (checkpoint, device, clip length, threshold, class names).
- `MODEL_C_*` - R(2+1)D (checkpoint, device, clip length, threshold).
- `MODEL_D_*` - ViViT (model dir, device, clip length, threshold, positive label).
- `MODEL_E_*` - VideoMAE (model dir, device, clip length, threshold, positive label).
- `HOST` / `PORT` / `DEBUG` - server settings.
- `CORS_ORIGINS` - allowed origins (`,` separated or `*`).

Any model with a valid checkpoint path will be initialized at startup; `model_manager` auto-selects the first available model. Device strings accept `cpu`, `cuda`, or `gpu` (gpu/cuda fall back to cpu if CUDA is unavailable). Keep the two `.env` files aligned when switching between Docker and local runs.

## Turning on GPU inference
1) Make sure the host sees your GPU:
```bash
nvidia-smi
```
2) Use CUDA-enabled PyTorch.
- Local: install from the CUDA wheel index, e.g.
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
- Docker: build with the CUDA index so the container gets GPU wheels while still installing other deps from PyPI:
```bash
PIP_INDEX_URL=https://download.pytorch.org/whl/cu121 PIP_EXTRA_INDEX_URL=https://pypi.org/simple docker compose build --no-cache backend
```
3) Set devices to CUDA in `.env` (root for Docker, `app/.env` for local):
```env
MODEL_A_DEVICE=cuda
MODEL_B_DEVICE=cuda
MODEL_C_DEVICE=cuda
MODEL_D_DEVICE=cuda
MODEL_E_DEVICE=cuda
```
4) If using Docker, allow GPU passthrough (Compose override example):
```yaml
# docker-compose.gpu.yml
services:
  backend:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
Run with:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
5) Verify the container sees the GPU (optional):
```bash
docker compose exec backend python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## API surface
- Live: `/live`, `/api/live/stream` (WebSocket), `/api/live/camera/stream` (MJPEG)
- Offline: `/offline`, `/api/offline/upload`, `/api/offline/status/{job_id}`, `/api/offline/result/{job_id}`
- Model control: `/api/model/select`, `/api/model/current`, `/api/model/load`

## Troubleshooting
- If no models load, confirm checkpoint paths in `.env` exist inside the container/local folder.
- CPU fallback happens automatically when CUDA is not available even if `*_DEVICE` is set to `cuda`/`gpu`.
- For slow builds with CUDA wheels, add `PIP_NO_CACHE_DIR=1` to keep images smaller.
