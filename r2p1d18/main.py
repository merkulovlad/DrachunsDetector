import json
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader, cpu
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

# --------------------
# Config — matches your training defaults
# --------------------
CLIP_LEN = int(os.getenv("CLIP_LEN", 16))
SIZE = int(os.getenv("SIZE", 112))
LABELS = ["NonFight", "Fight"]  # 0 -> NonFight, 1 -> Fight
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "best_r2p1d18.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------
# Utility: preprocessing and model loading
# --------------------
def _uniform_indices(n: int, T: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((T,), dtype=np.int64)
    if n <= T:
        return np.linspace(0, n - 1, T).astype(np.int64)
    return np.linspace(0, n - 1, T).astype(np.int64)


def _load_video_tensor(path: str, T: int = CLIP_LEN, size: int = SIZE) -> torch.Tensor:
    """
    Load a video file with Decord and convert to normalized tensor (C,T,H,W).
    """
    vr = VideoReader(path, ctx=cpu(0))
    idx = _uniform_indices(len(vr), T)
    frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3) RGB uint8

    frames = np.stack(
        [cv2.resize(fr, (size, size), interpolation=cv2.INTER_LINEAR) for fr in frames],
        axis=0,
    )  # (T, S, S, 3)

    x = torch.from_numpy(frames).float().div(255.0).permute(3, 0, 1, 2)  # (3,T,S,S)
    mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)
    x = (x - mean) / std
    return x


def _frames_to_tensor(frames: List[np.ndarray], size: int = SIZE) -> torch.Tensor:
    """
    Frames: list of RGB uint8 frames, shape (H, W, 3).
    Return normalized (C,T,H,W).
    """
    arr = np.stack(
        [cv2.resize(fr, (size, size), interpolation=cv2.INTER_LINEAR) for fr in frames],
        axis=0,
    )  # (T,S,S,3)
    x = torch.from_numpy(arr).float().div(255.0).permute(3, 0, 1, 2)
    mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)
    return (x - mean) / std


def _strip_module_prefix(state_dict):
    # Only strip when the key actually starts with 'module.'
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def load_model(weights_path: str) -> nn.Module:
    # base arch
    weights_enum = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights_enum)
    model.fc = nn.Linear(model.fc.in_features, 2)  # your 2-class head

    raw = torch.load(weights_path, map_location="cpu")

    # Try to find the actual tensor dict
    cand = None
    if isinstance(raw, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in raw and isinstance(raw[k], dict):
                cand = raw[k]
                break
        if cand is None:
            # Maybe it's already the state_dict
            cand = raw
    else:
        # Sometimes people torch.save(model.state_dict()) directly (dict),
        # but if not a dict, it's an unsupported format
        raise RuntimeError("Checkpoint format not recognized (expected dict)")

    # Clean any 'module.' prefixes from DataParallel
    cand = _strip_module_prefix(cand)

    # (Optional) print a couple keys if you want to sanity-check
    # print("Some keys:", list(cand.keys())[:10])

    # Load strictly; if you still see Unexpected/ Missing keys, set strict=False to diagnose
    missing, unexpected = model.load_state_dict(cand, strict=False)
    if missing or unexpected:
        # Helpful diagnostics
        print(">> load_state_dict diagnostics")
        if missing:
            print("  Missing keys:", missing[:20], "..." if len(missing) > 20 else "")
        if unexpected:
            print(
                "  Unexpected keys:",
                unexpected[:20],
                "..." if len(unexpected) > 20 else "",
            )
        # If ONLY the classifier head mismatches (e.g., different fc size), that would show up here.
        # Since we set model.fc = 2, you should be fine.

    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def predict_logits(model: nn.Module, x_3tHW: torch.Tensor) -> torch.Tensor:
    x = x_3tHW.unsqueeze(0).to(DEVICE, non_blocking=True)  # (1,3,T,H,W)
    logits = model(x)  # (1,2)
    return logits.squeeze(0).cpu()


def format_response(logits: torch.Tensor) -> Dict:
    probs = torch.softmax(logits, dim=-1).tolist()
    top_idx = int(torch.argmax(logits).item())
    return {
        "prediction": LABELS[top_idx],
        "probabilities": {LABELS[i]: float(p) for i, p in enumerate(probs)},
    }


def _jpeg_bytes_to_rgb(byte_data: bytes) -> np.ndarray:
    arr = np.frombuffer(byte_data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="R2Plus1D-18 Video Classifier", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

MODEL: Optional[nn.Module] = None


@app.on_event("startup")
def _startup():
    global MODEL
    weights_file = Path(MODEL_WEIGHTS)
    if not weights_file.exists():
        raise RuntimeError(f"Model weights not found: {weights_file.resolve()}")
    MODEL = load_model(str(weights_file))


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


# ---------- Single-result endpoints ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Multipart upload -> one prediction for the whole file.
    """
    try:
        suffix = Path(file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in iter(lambda: file.file.read(1024 * 1024), b""):
                tmp.write(chunk)
            tmp_path = tmp.name

        x = _load_video_tensor(tmp_path, T=CLIP_LEN, size=SIZE)
        logits = predict_logits(MODEL, x)
        return JSONResponse(format_response(logits))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process video: {e}")
    finally:
        try:
            if "tmp_path" in locals() and Path(tmp_path).exists():
                Path(tmp_path).unlink()
        except Exception:
            pass


@app.post("/predict-stream")
async def predict_stream(request: Request):
    """
    application/octet-stream -> server buffers to a temp file -> predict once.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            async for chunk in request.stream():
                if not chunk:
                    break
                tmp.write(chunk)
            tmp_path = tmp.name

        x = _load_video_tensor(tmp_path, T=CLIP_LEN, size=SIZE)
        logits = predict_logits(MODEL, x)
        return JSONResponse(format_response(logits))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process streamed video: {e}"
        )
    finally:
        try:
            if "tmp_path" in locals() and Path(tmp_path).exists():
                Path(tmp_path).unlink()
        except Exception:
            pass


# ---------- Realtime (SSE): webcam/RTSP/file ----------
@app.get("/predict-rt")
def predict_realtime_sse(
    source: str = Query(
        ..., description="Camera index ('0') or RTSP/HTTP URL or file path"
    ),
    stride_frames: int = Query(
        8, ge=1, description="Emit one prediction every N frames"
    ),
    realtime: int = Query(0, ge=0, le=1, description="If 1, pace reads to ~source FPS"),
    max_events: int = Query(
        0, ge=0, description="0=unlimited; else stop after N emissions"
    ),
):
    """
    Streams predictions as Server-Sent Events (text/event-stream).
    - source=0            -> local webcam
    - source=rtsp://...   -> IP camera
    - source=/path/file   -> video file (optionally throttled to 'realtime')
    """
    cap = (
        cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    )
    if not cap.isOpened():
        raise HTTPException(
            status_code=400, detail=f"Cannot open video source: {source}"
        )

    buf = deque(maxlen=CLIP_LEN)
    events_sent = 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_idx = 0

    frame_period = (1.0 / fps) if (realtime == 1 and fps > 0) else 0.0
    last_time = time.time()

    def gen():
        nonlocal events_sent, frame_idx, last_time
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break

                # pace to source FPS if requested
                if frame_period > 0:
                    now = time.time()
                    sleep_for = (last_time + frame_period) - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    last_time = max(last_time + frame_period, time.time())

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                buf.append(rgb)
                frame_idx += 1

                if len(buf) == CLIP_LEN and (frame_idx % stride_frames == 0):
                    x = _frames_to_tensor(list(buf), SIZE)
                    logits = predict_logits(MODEL, x)
                    result = format_response(logits)
                    payload = {
                        "t": time.time(),
                        "frame": frame_idx,
                        "fps": fps,
                        **result,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    events_sent += 1
                    if max_events and events_sent >= max_events:
                        break
        finally:
            cap.release()

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- Realtime (WebSocket): browser pushes JPEG frames ----------
@app.websocket("/ws-frames")
async def ws_frames(ws: WebSocket):
    """
    Browser workflow:
      1) ws.send(JSON.stringify({stride_frames: 8}))    # optional config text
      2) Then repeatedly send binary JPEG frames (Blob/ArrayBuffer)
      3) Server replies with JSON predictions periodically
    """
    await ws.accept()
    buf = deque(maxlen=CLIP_LEN)
    stride_frames = 8
    frame_idx = 0

    try:
        # Optional first text message with config
        first = await ws.receive()
        if (
            first["type"] == "websocket.receive"
            and "text" in first
            and first["text"] is not None
        ):
            try:
                cfg = json.loads(first["text"])
                stride_frames = int(cfg.get("stride_frames", stride_frames))
            except Exception:
                pass
        else:
            if first.get("bytes"):
                rgb = _jpeg_bytes_to_rgb(first["bytes"])
                buf.append(rgb)
                frame_idx += 1

        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                rgb = _jpeg_bytes_to_rgb(msg["bytes"])
                buf.append(rgb)
                frame_idx += 1

                if len(buf) == CLIP_LEN and (frame_idx % stride_frames == 0):
                    x = _frames_to_tensor(list(buf), SIZE)
                    logits = predict_logits(MODEL, x)
                    result = format_response(logits)
                    await ws.send_json({"frame": frame_idx, **result})

            elif "text" in msg and msg["text"] is not None:
                # could live-update config if needed
                pass
            else:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        await ws.close()


# --- Offline analyze and return matplotlib PNG ---
import base64
import io

import matplotlib
from fastapi import Form

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sliding_window_probs(
    vpath: str, stride_frames: int, T: int
) -> (List[int], List[float], float):
    """
    Read video with decord, keep latest T frames, infer every 'stride_frames'.
    Returns: (frame_indices, p_fight_list, fps)
    """
    vr = VideoReader(vpath, ctx=cpu(0))
    n = len(vr)
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 0.0

    buf = []
    frame_indices = []
    p_fight = []

    for i in range(n):
        fr = vr[i].asnumpy()  # RGB uint8
        buf.append(fr)
        if len(buf) > T:
            buf.pop(0)

        if len(buf) == T and ((i + 1) % stride_frames == 0):
            x = _frames_to_tensor(buf, SIZE)  # (3,T,H,W)
            logits = predict_logits(MODEL, x)
            probs = torch.softmax(logits, dim=-1).tolist()
            p_fight.append(float(probs[1]))
            frame_indices.append(i + 1)  # end frame of this window

    return frame_indices, p_fight, fps


def _plot_probs_png_base64(frames: List[int], pf: List[float], thr: float) -> str:
    fig = plt.figure(figsize=(8, 3))
    ax = plt.gca()
    ax.plot(frames, pf, linewidth=1.8)
    ax.axhline(thr, linestyle="--", linewidth=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Frame")
    ax.set_ylabel("P(Fight)")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    stride_frames: int = Form(8),
    threshold: float = Form(0.5),
):
    """
    Upload a video and get:
      - sliding-window predictions (frames[], p_fight[])
      - matplotlib PNG as base64
      - frames crossing threshold
    """
    try:
        suffix = Path(file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in iter(lambda: file.file.read(1024 * 1024), b""):
                tmp.write(chunk)
            tmp_path = tmp.name

        frames, p_fight, fps = _sliding_window_probs(tmp_path, stride_frames, CLIP_LEN)

        if not frames:
            raise HTTPException(
                status_code=400,
                detail="Video too short for one window; try smaller CLIP_LEN or stride_frames.",
            )

        # find frames where prob >= threshold
        fight_frames = [int(fr) for fr, p in zip(frames, p_fight) if p >= threshold]

        pngb64 = _plot_probs_png_base64(frames, p_fight, threshold)

        return JSONResponse(
            {
                "frames": frames,
                "p_fight": p_fight,
                "fps": fps,
                "stride_frames": stride_frames,
                "threshold": threshold,
                "fight_frames": fight_frames,
                "plot_png_b64": pngb64,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analyze failed: {e}")
    finally:
        try:
            if "tmp_path" in locals() and Path(tmp_path).exists():
                Path(tmp_path).unlink()
        except Exception:
            pass
