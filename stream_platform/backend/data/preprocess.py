import torch
import torchvision.transforms.v2 as T
import cv2

frame_tf = T.Compose([
    T.ToImage(),
    T.ConvertImageDtype(torch.float32),
    T.Resize((200, 200)),
    T.CenterCrop((172, 172))
    # Note: Training uses ConvertImageDtype(torch.float32) which keeps [0,255] range
    # No normalization to [0,1] - this was the issue!
])

def preprocess_frame(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)        # H,W,C (np uint8)
    x = frame_tf(rgb)                                 # -> C,H,W (float32, [0,255])
    return x
    
