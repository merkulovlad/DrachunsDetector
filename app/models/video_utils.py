"""Utility helpers for video-based classifiers."""
from typing import Iterable, Sequence

import torch


def prepare_clip_tensor(clip: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """
    Ensure the input clip matches the `(B, C, T, H, W)` layout expected by
    most video backbones.

    Accepts either a single tensor or an iterable of frame tensors shaped `(C, H, W)`.
    """
    if isinstance(clip, torch.Tensor):
        clip_tensor = clip
    else:
        clip_tensor = torch.stack(list(clip), dim=0)  # (T, C, H, W)

    if clip_tensor.ndim == 4:
        if clip_tensor.shape[0] in (1, 3):
            pass  # Already (C, T, H, W) or (C, H, W)
        elif clip_tensor.shape[1] in (1, 3):
            clip_tensor = clip_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        elif clip_tensor.shape[-1] in (1, 3):
            clip_tensor = clip_tensor.permute(-1, 0, 1, 2)  # (C, T, H, W)
        else:
            raise ValueError(f"Cannot infer channel dimension for clip shape {clip_tensor.shape}")
        if clip_tensor.ndim == 4:
            clip_tensor = clip_tensor.unsqueeze(0)
    elif clip_tensor.ndim == 5:
        if clip_tensor.shape[1] not in (1, 3):
            channel_axis = next((i for i, s in enumerate(clip_tensor.shape) if s in (1, 3)), None)
            if channel_axis is None:
                raise ValueError(f"Cannot infer channel dimension for clip shape {clip_tensor.shape}")
            perm = [0, channel_axis] + [i for i in range(1, clip_tensor.ndim) if i != channel_axis]
            clip_tensor = clip_tensor.permute(*perm)
    else:
        raise ValueError(f"Unsupported clip shape {clip_tensor.shape}")

    return clip_tensor.contiguous()
