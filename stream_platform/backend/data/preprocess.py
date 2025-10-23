import torch
import torchvision.transforms.v2 as T
import cv2

frame_tf = T.Compose([
    T.ToImage(),
    T.ConvertImageDtype(torch.float32),
    T.Resize(172),
    T.CenterCrop(172),
    T.Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
])

def preprocess_frame(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = frame_tf(rgb)
    return x
    
