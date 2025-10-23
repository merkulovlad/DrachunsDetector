from email.policy import strict
from movinets.config import _C
from movinets import MoViNet
import os

import torch
from stream_platform.backend.utils.logger import get_logger

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "epoch009.pt")
MODEL = MoViNet(_C.MODEL.MoViNetA0, causal = True, pretrained = False )

def _detect_device() -> str:
    # Force CPU for debugging
    return "cpu"

device = _detect_device()
log = get_logger("movinet")

def download_model(model=MODEL, ckpt_path: str = CHECKPOINT_PATH):
    # Dummy implementation of model download
    if not os.path.exists(ckpt_path):
        log.error(f"cannot find model at {ckpt_path}")
        return None
    model.to(device)
    log.info(f"Model {model} downloaded successfully")
    log.info("loading_checkpoint", path=ckpt_path, device=str(device), strict=strict)
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = {}

    # extract state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    else:
        state = ckpt  # plain state_dict

    # handle DataParallel-style keys ("module.")
    if any(k.startswith("module.") for k in state.keys()):
        log.info("stripping_dataparallel_prefix")
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # Ensure strict is a bool (avoid passing email.policy.strict by accident)
    load_strict = True if strict is None else bool(strict)
    missing, unexpected = model.load_state_dict(state, strict=load_strict)
    if missing or unexpected:
        log.warning("state_dict_mismatch", missing=missing, unexpected=unexpected, strict=strict)

    model.eval()
    # for streaming MoViNet it's often good to clear buffers once before inference
    if hasattr(model, "clean_activation_buffers"):
        model.clean_activation_buffers()

    log.info("checkpoint_loaded", missing=len(missing), unexpected=len(unexpected))
    return model, meta
