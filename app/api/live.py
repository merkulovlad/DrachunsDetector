"""Live streaming API routes."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import asyncio
from typing import Optional
import logging

from core.model_manager import model_manager, ModelType, CHECKPOINT_FIELDS
from core.config import settings

router = APIRouter(prefix="/api/live", tags=["live"])
logger = logging.getLogger(__name__)

# Active camera stream
active_camera = None
camera_lock = asyncio.Lock()


def _resolve_model_key(requested: Optional[str]) -> Optional[ModelType]:
    available = model_manager.list_available_models()
    if not available:
        return None
    if requested in available:
        return requested  # type: ignore
    current = model_manager.get_current_model_type()
    if current in available:
        return current
    return available[0]


def _prepare_detector(model_type: Optional[ModelType]):
    if model_type is None:
        return None, False
    try:
        detector = model_manager.get_model(model_type)
        detector.reset()
        return detector, True
    except Exception as exc:
        logger.info(f"Model {model_type} not initialized yet: {exc}")

    field_name = CHECKPOINT_FIELDS.get(model_type)
    checkpoint = getattr(settings, field_name, "") if field_name else ""
    if not checkpoint:
        return None, False

    try:
        model_manager.initialize_models(**{field_name: checkpoint})
        detector = model_manager.get_model(model_type)
        detector.reset()
        logger.info(f"Auto-loaded model {model_type} from {checkpoint}")
        return detector, True
    except Exception as exc:
        logger.warning(f"Failed to auto-load model {model_type}: {exc}")
        return None, False


class CameraStream:
    """Manage camera stream."""
    
    def __init__(self, source: int = 0):
        self.source = source
        self.cap = None
        self.is_running = False
    
    def start(self):
        """Start camera capture."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source)
            self.is_running = True
            return True
        return False
    
    def stop(self):
        """Stop camera capture."""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def read_frame(self):
        """Read a frame from camera."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None


@router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    model: Optional[str] = Query(None, description="Model key to use (a, b, c, ...)")
):
    """
    WebSocket endpoint for real-time video streaming.
    
    Client sends video frames, server responds with annotated frames.
    """
    await websocket.accept()
    
    model_type = _resolve_model_key(model)
    detector, model_ready = _prepare_detector(model_type)
    model_label = model_type or "none"
    if model_ready:
        logger.info(f"WebSocket stream started with model {model_label}")
    else:
        logger.warning("WebSocket stream running without an initialized model")

    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_bytes()

            # Decode frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            if model_ready and detector is not None:
                # Process frame through the selected model
                try:
                    annotated, prob, _ = detector.process_frame(frame)
                except Exception as err:
                    logger.error(f"Error during model processing: {err}")
                    # fallback to original frame with error overlay
                    annotated = frame.copy()
                    cv2.putText(annotated, "MODEL ERROR", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            else:
                # Passthrough: echo original frame with a small overlay indicating no model
                annotated = frame.copy()
                cv2.putText(annotated, f"NO MODEL ({model_label})", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            # Encode and send back
            _, buffer = cv2.imencode('.jpg', annotated)
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for model {model_label}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/camera/start")
async def start_camera(source: int = Query(0, description="Camera source index")):
    """Start camera capture."""
    global active_camera
    
    async with camera_lock:
        if active_camera is None:
            active_camera = CameraStream(source)
            active_camera.start()
            return {"status": "started", "source": source}
        else:
            return {"status": "already_running", "source": active_camera.source}


@router.get("/camera/stop")
async def stop_camera():
    """Stop camera capture."""
    global active_camera
    
    async with camera_lock:
        if active_camera:
            active_camera.stop()
            active_camera = None
            return {"status": "stopped"}
        else:
            return {"status": "not_running"}


@router.get("/camera/stream")
async def camera_stream(model: Optional[str] = Query(None, description="Model key to use (a, b, c, ...)")):
    """
    MJPEG stream endpoint for camera feed.
    
    Returns an MJPEG stream of processed frames.
    """
    global active_camera
    
    model_type = _resolve_model_key(model)
    detector, ready = _prepare_detector(model_type)
    if model_type is None:
        return {"error": "No models available"}
    if not ready or detector is None:
        return {"error": f"Model {model_type} not available"}
    
    async def generate():
        """Generate MJPEG frames."""
        if active_camera is None or not active_camera.is_running:
            # Return empty frame
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(empty, "No camera active", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', empty)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
        
        while active_camera and active_camera.is_running:
            frame = active_camera.read_frame()
            
            if frame is None:
                await asyncio.sleep(0.033)  # ~30 FPS
                continue
            
            # Process frame
            annotated, prob, _ = detector.process_frame(frame)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            await asyncio.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/status")
async def get_status():
    """Get current stream status."""
    return {
        "camera_active": active_camera is not None and active_camera.is_running,
        "current_model": model_manager.get_current_model_type(),
        "available_models": model_manager.list_available_models(),
    }
