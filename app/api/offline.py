"""Offline video processing API routes."""
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import cv2
from typing import Dict, Optional
import logging
from datetime import datetime

from core.config import settings
from core.model_manager import model_manager, ModelType

router = APIRouter(prefix="/api/offline", tags=["offline"])
logger = logging.getLogger(__name__)

# Job storage
jobs: Dict[str, dict] = {}


def _resolve_model_choice(requested: Optional[str]) -> Optional[ModelType]:
    available = model_manager.list_available_models()
    if not available:
        return None
    if requested in available:
        return requested  # type: ignore
    current = model_manager.get_current_model_type()
    if current in available:
        return current
    return available[0]


def process_video_job(
    job_id: str,
    input_path: Path,
    output_path: Path,
    model_type: ModelType
):
    """
    Background task to process video.
    
    Args:
        job_id: Unique job identifier
        input_path: Path to input video
        output_path: Path to save output video
        model_type: Model to use for processing
    """
    try:
        # Update job status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        # Get model
        detector = model_manager.get_model(model_type)
        detector.reset()
        
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError("Could not open input video")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        jobs[job_id]["total_frames"] = total_frames
        jobs[job_id]["fps"] = fps
        
        # Create output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        violence_detections = []
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated, prob, _ = detector.process_frame(frame)
            
            # Write frame
            out.write(annotated)
            
            # Track violence detections
            if prob >= detector.threshold:
                violence_detections.append({
                    "frame": frame_count,
                    "timestamp": frame_count / fps,
                    "probability": prob
                })
            
            frame_count += 1
            
            # Update progress
            jobs[job_id]["processed_frames"] = frame_count
            jobs[job_id]["progress"] = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        
        # Cleanup
        cap.release()
        out.release()
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["output_path"] = str(output_path)
        jobs[job_id]["violence_detections"] = violence_detections
        jobs[job_id]["violence_detected"] = len(violence_detections) > 0
        jobs[job_id]["progress"] = 100
        
        logger.info(f"Job {job_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: Optional[str] = Query(None, description="Model key to use (a, b, c, ...)")
):
    """
    Upload a video file for offline processing.
    
    Returns a job ID for tracking progress.
    """
    # Validate model
    model_type = _resolve_model_choice(model)
    if model_type is None:
        raise HTTPException(status_code=400, detail="No models available")
    
    if model_type not in model_manager.list_available_models():
        raise HTTPException(status_code=400, detail=f"Model {model_type} not available")
    
    # Validate file
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    input_path = settings.uploads_dir / f"{job_id}_{file.filename}"
    output_path = settings.outputs_dir / f"{job_id}_output.mp4"
    
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create job record
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "model": model_type,
        "filename": file.filename,
        "input_path": str(input_path),
        "created_at": datetime.now().isoformat(),
        "processed_frames": 0,
        "total_frames": 0,
        "progress": 0,
    }
    
    # Start background processing
    background_tasks.add_task(
        process_video_job,
        job_id,
        input_path,
        output_path,
        model_type
    )
    
    logger.info(f"Created job {job_id} for file {file.filename} with model {model_type}")
    
    return {
        "job_id": job_id,
        "status": "queued",
        "model": model_type
    }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "model": job["model"],
        "filename": job["filename"],
        "progress": job.get("progress", 0),
        "processed_frames": job.get("processed_frames", 0),
        "total_frames": job.get("total_frames", 0),
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
    }


@router.get("/result/{job_id}")
async def get_job_result(job_id: str):
    """Get the detailed results of a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is {job['status']}, not completed")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "model": job["model"],
        "filename": job["filename"],
        "total_frames": job.get("total_frames", 0),
        "fps": job.get("fps", 0),
        "violence_detected": job.get("violence_detected", False),
        "violence_detections": job.get("violence_detections", []),
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "download_url": f"/api/offline/download/{job_id}"
    }


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download the processed video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is {job['status']}, not completed")
    
    output_path = Path(job["output_path"])
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"processed_{job['filename']}"
    )


@router.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return {
        "jobs": list(jobs.values())
    }


@router.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Delete files
    if "input_path" in job:
        input_path = Path(job["input_path"])
        if input_path.exists():
            input_path.unlink()
    
    if "output_path" in job:
        output_path = Path(job["output_path"])
        if output_path.exists():
            output_path.unlink()
    
    # Remove job
    del jobs[job_id]
    
    return {"status": "deleted", "job_id": job_id}
