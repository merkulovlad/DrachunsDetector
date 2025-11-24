"""FastAPI application entry point."""
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))  # allow `core`/`models` when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "main_code"))
sys.path.insert(0, str(Path(__file__).parent.parent / "stream_platform"))

from core.config import settings
from core.model_manager import model_manager, CHECKPOINT_FIELDS
from api import live, offline

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize models if checkpoints are set
    model_manager.initialize_models()
    
    available = model_manager.list_available_models()
    logger.info(f"Available models: {available}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(settings.base_dir / "static")), name="static")

# Templates
templates = Jinja2Templates(directory=str(settings.base_dir / "templates"))

# Include routers
app.include_router(live.router)
app.include_router(offline.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "endpoints": {
            "live_monitor": "/live",
            "offline_analyzer": "/offline",
            "api_docs": "/docs"
        }
    }


@app.get("/live")
async def live_page(request: Request):
    """Live monitoring page."""
    return templates.TemplateResponse(
        "live.html",
        {
            "request": request,
            "available_models": model_manager.list_available_models(),
            "current_model": model_manager.get_current_model_type()
        }
    )


@app.get("/offline")
async def offline_page(request: Request):
    """Offline analysis page."""
    return templates.TemplateResponse(
        "offline.html",
        {
            "request": request,
            "available_models": model_manager.list_available_models()
        }
    )


@app.post("/api/model/select")
async def select_model(payload: dict):
    """Select active model. Accepts JSON body: {"model": "a"}"""
    model = payload.get("model") if isinstance(payload, dict) else None
    available = model_manager.list_available_models()
    if not available:
        return {"error": "No models initialized", "status": "error"}
    if model not in available:
        return {
            "error": f"Invalid model. Choose one of {', '.join(m.upper() for m in available)}",
            "status": "error",
        }

    try:
        model_manager.set_current_model(model)
        return {
            "status": "ok",
            "model": model,
            "current": model_manager.get_current_model_type()
        }
    except ValueError as e:
        return {"error": str(e), "status": "error"}


@app.post("/api/model/load")
async def load_model(payload: dict):
    """Load a model at runtime. JSON body: {"model": "a"|"b", "checkpoint": "/path/to.ckpt"}
    If checkpoint omitted, will attempt to load from configured .env paths.
    """
    model = payload.get("model") if isinstance(payload, dict) else None
    checkpoint = payload.get("checkpoint") if isinstance(payload, dict) else None
    field_name = CHECKPOINT_FIELDS.get(model)
    if field_name is None:
        return {"error": "Invalid model key"}

    if checkpoint:
        path_obj = Path(checkpoint)
        if not path_obj.exists():
            return {"error": f"Checkpoint not found: {checkpoint}"}
        setattr(settings, field_name, str(path_obj.resolve()))

    try:
        model_manager.initialize_models(**{field_name: getattr(settings, field_name)})

        available = model_manager.list_available_models()
        return {"status": "ok", "available": available}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/model/current")
async def get_current_model():
    """Get current model."""
    return {
        "current": model_manager.get_current_model_type(),
        "available": model_manager.list_available_models()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
