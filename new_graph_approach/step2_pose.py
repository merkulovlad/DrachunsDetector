
import torch
import numpy as np
from typing import Iterable, List
from ultralytics import YOLO


class PoseEstimator:
    def __init__(self, weights="yolov8n-pose.pt", imgsz=640, conf=0.25, iou=0.5, device="auto", max_det=50):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.max_det = max_det

    def _result_to_detections(self, res) -> List[tuple]:
        detections = []
        if res.boxes is None or res.keypoints is None:
            return detections
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
        keypoints = res.keypoints.data.cpu().numpy()
        for bbox, score, cls, kp in zip(boxes, scores, classes, keypoints):
            if int(cls) != 0:  # keep only 'person' class
                continue
            detections.append(
                (bbox.astype(float), float(score), int(cls), kp[:, :3].astype(float))
            )
        return detections

    @torch.inference_mode()
    def infer(self, frame):
        # returns list of detections: [ (xyxy, score, cls, keypoints(17x3)) , ...]
        res = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )[0]
        return self._result_to_detections(res)

    @torch.inference_mode()
    def infer_batch(self, frames: Iterable[np.ndarray]) -> List[List[tuple]]:
        frames = list(frames)
        if len(frames) == 0:
            return []
        results = self.model.predict(
            source=frames,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        return [self._result_to_detections(res) for res in results]
