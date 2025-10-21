import torch
import torch.nn.functional as F

def _detect_device() -> str:
    # Force CPU for debugging
    return "cpu"

DEVICE = _detect_device()

def _prepare_clip_tensor(clip):
    """
    Ensure the input clip matches MoViNet expected shape: (B, C, T, H, W).
    Accepts a tensor or an iterable of frame tensors (C, H, W).
    """
    if isinstance(clip, torch.Tensor):
        clip_tensor = clip
    else:
        clip_tensor = torch.stack(list(clip), dim=0)  # T, C, H, W

    if clip_tensor.ndim == 4:
        if clip_tensor.shape[0] in (1, 3):
            # Already (C, T, H, W) or (C, H, W); nothing to permute
            pass
        elif clip_tensor.shape[1] in (1, 3):
            # (T, C, H, W) -> (C, T, H, W)
            clip_tensor = clip_tensor.permute(1, 0, 2, 3)
        elif clip_tensor.shape[-1] in (1, 3):
            # (T, H, W, C) -> (C, T, H, W)
            clip_tensor = clip_tensor.permute(-1, 0, 1, 2)
        else:
            raise ValueError(f"Cannot infer channel dimension for clip shape {clip_tensor.shape}")
        if clip_tensor.ndim == 4:
            clip_tensor = clip_tensor.unsqueeze(0)
    elif clip_tensor.ndim == 5:
        # If channels axis is not index 1, move it there
        if clip_tensor.shape[1] not in (1, 3):
            channel_axis = next((i for i, s in enumerate(clip_tensor.shape) if s in (1, 3)), None)
            if channel_axis is None:
                raise ValueError(f"Cannot infer channel dimension for clip shape {clip_tensor.shape}")
            perm = [0, channel_axis] + [i for i in range(1, clip_tensor.ndim) if i != channel_axis]
            clip_tensor = clip_tensor.permute(*perm)
    else:
        raise ValueError(f"Unsupported clip shape {clip_tensor.shape}")

    return clip_tensor.contiguous()


@torch.inference_mode()
def run_inference(model: torch.nn.Module, clip, autocast=True, device=DEVICE, return_logits=False):
    clip_tensor = _prepare_clip_tensor(clip).to(device, non_blocking=True)
    use_autocast = autocast and device == "cuda"
    with torch.autocast(device_type=("cuda" if device == "cuda" else "cpu"), enabled=use_autocast):
        logits = model(clip_tensor)
    
    if return_logits:
        return logits
    else:
        probs = F.softmax(logits, dim=1)
        return probs
