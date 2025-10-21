from movinets import MoViNet
from movinets.config import _C

import torch.nn as nn

def build_movinet_a0_stream(num_classes: int, *, pretrained: bool = False):
    """
    Build the same architecture used in training (A0, causal=True).
    Use pretrained=False when loading your own fine-tuned weights.
    """
    model = MoViNet(
        _C.MODEL.MoViNetA0,
        causal=True,
        pretrained=pretrained,
        num_classes=num_classes,
        conv_type="2plus1d",
        tf_like=True,
    )
    model.classifier[3] = nn.Conv3d(2048, num_classes, (1,1,1))
    return model
