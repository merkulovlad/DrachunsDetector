"""
Robust Pose Detection Pipeline
Replaces MoveNet with YOLO + MediaPipe for better performance on poor quality video
Maintains compatibility with existing STGCN pipeline
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import time
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class RobustPoseDetector:
    """
    Robust pose detection using YOLO for person detection + MediaPipe for pose estimation
    Handles unlimited people and works better on poor quality video
    """
    
    def __init__(self, 
                 yolo_model_path: str = 'yolov8n.pt',
                 max_people: int = 10,
                 confidence_threshold: float = 0.5,
                 pose_confidence: float = 0.5):
        """
        Args:
            yolo_model_path: Path to YOLO model (yolov8n.pt, yolov8s.pt, etc.)
            max_people: Maximum number of people to track
            confidence_threshold: YOLO detection confidence threshold
            pose_confidence: MediaPipe pose confidence threshold
        """
        self.max_people = max_people
        self.confidence_threshold = confidence_threshold
        self.pose_confidence = pose_confidence
        
        # Initialize YOLO for person detection
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose_estimator = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0=Light, 1=Medium, 2=Heavy
            smooth_landmarks=True,
            min_detection_confidence=pose_confidence,
            min_tracking_confidence=pose_confidence
        )
        
        # For tracking people across frames
        self.tracked_people = deque(maxlen=max_people)
        self.next_person_id = 0
        
        print("✅ Robust Pose Detector initialized")
    
    def detect_people(self, frame: np.ndarray) -> List[dict]:
        """
        Detect people in frame using YOLO
        Returns list of person bounding boxes with confidence scores
        """
        results = self.yolo_model(frame, classes=[0], conf=self.confidence_threshold, verbose=False)
        
        people = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                people.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                })
        
        return people
    
    def estimate_pose(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Estimate pose for a single person using MediaPipe
        Returns pose landmarks in format compatible with existing pipeline
        """
        x1, y1, x2, y2 = bbox
        
        # Add padding to bbox
        h, w = frame.shape[:2]
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None
        
        # Convert to RGB for MediaPipe
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Estimate pose
        results = self.pose_estimator.process(rgb_crop)
        
        if results.pose_landmarks is None:
            return None
        
        # Convert MediaPipe landmarks to our format
        landmarks = results.pose_landmarks.landmark
        
        # Convert to numpy array with same format as MoveNet
        pose_data = np.zeros((17, 3), dtype=np.float32)
        
        # MediaPipe pose landmark mapping to COCO format
        mp_to_coco = {
            0: 0,   # nose
            2: 1,   # left_eye_inner
            5: 2,   # right_eye_inner
            7: 3,   # left_ear
            8: 4,   # right_ear
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10, # right_wrist
            23: 11, # left_hip
            24: 12, # right_hip
            25: 13, # left_knee
            26: 14, # right_knee
            27: 15, # left_ankle
            28: 16  # right_ankle
        }
        
        for mp_idx, coco_idx in mp_to_coco.items():
            landmark = landmarks[mp_idx]
            # Convert relative coordinates to absolute pixel coordinates
            pose_data[coco_idx, 0] = landmark.x * (x2 - x1) + x1  # x coordinate
            pose_data[coco_idx, 1] = landmark.y * (y2 - y1) + y1  # y coordinate
            pose_data[coco_idx, 2] = landmark.visibility  # confidence
        
        return pose_data
    
    def track_people(self, current_people: List[dict]) -> List[dict]:
        """
        Track people across frames using simple distance-based tracking
        """
        tracked = []
        
        for person in current_people:
            person_center = person['center']
            best_match = None
            best_distance = float('inf')
            
            # Find closest tracked person
            for tracked_person in self.tracked_people:
                distance = np.linalg.norm(np.array(person_center) - np.array(tracked_person['center']))
                if distance < best_distance and distance < 100:  # max distance threshold
                    best_distance = distance
                    best_match = tracked_person
            
            if best_match is not None:
                # Update existing person
                best_match.update(person)
                tracked.append(best_match)
            else:
                # New person
                person['id'] = self.next_person_id
                self.next_person_id += 1
                tracked.append(person)
        
        # Update tracked people
        self.tracked_people = deque(tracked, maxlen=self.max_people)
        return tracked
    
    def detect_poses(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Main detection function - detects people and estimates poses
        Returns list of pose arrays compatible with existing pipeline
        """
        # Detect people
        people = self.detect_people(frame)
        
        # Track people across frames
        tracked_people = self.track_people(people)
        
        # Estimate poses for each tracked person
        poses = []
        for person in tracked_people:
            pose = self.estimate_pose(frame, person['bbox'])
            if pose is not None:
                poses.append(pose)
        
        return poses
    
    def process_frame(self, frame: np.ndarray, prev_poses: List[np.ndarray], fps: float = 30.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process single frame and extract features compatible with existing pipeline
        Returns features in same format as MoveNet: (17, 5) per person
        """
        h, w = frame.shape[:2]
        
        # Detect poses
        current_poses = self.detect_poses(frame)
        
        # Extract features for each pose
        features_out = []
        
        for pose in current_poses:
            # Normalize coordinates to [-1, 1]
            x_norm = (pose[:, 0] / w) * 2 - 1
            y_norm = (pose[:, 1] / h) * 2 - 1
            conf = pose[:, 2]
            
            # Compute velocities
            velocities = np.zeros((17, 2))
            if prev_poses:
                # Find closest previous pose
                best_prev = None
                best_distance = float('inf')
                
                for prev_pose in prev_poses:
                    # Calculate average distance between corresponding joints
                    distances = np.sqrt((pose[:, 0] - prev_pose[:, 0])**2 + (pose[:, 1] - prev_pose[:, 1])**2)
                    avg_distance = np.mean(distances[conf > 0.3])  # Only consider confident joints
                    
                    if avg_distance < best_distance:
                        best_distance = avg_distance
                        best_prev = prev_pose
                
                if best_prev is not None:
                    dt = 1.0 / fps
                    for j in range(17):
                        if conf[j] > 0.3 and best_prev[j, 2] > 0.3:
                            vx = (pose[j, 0] - best_prev[j, 0]) / dt
                            vy = (pose[j, 1] - best_prev[j, 1]) / dt
                            # Normalize velocity
                            velocities[j, 0] = vx / w
                            velocities[j, 1] = vy / h
            
            # Create feature vector (same format as MoveNet)
            person_features = np.stack([
                x_norm, y_norm, conf,
                velocities[:, 0], velocities[:, 1]
            ], axis=-1)  # (17, 5)
            
            features_out.append(person_features)
        
        return features_out, current_poses
    
    def video_to_numpy(self, video_path: str, fps_default: float = 30.0) -> np.ndarray:
        """
        Convert video to numpy array compatible with existing pipeline
        Returns array of shape (T, P, V, F) where:
        - T: time frames
        - P: people (padded to max_people)
        - V: joints (17)
        - F: features (5: x, y, conf, vx, vy)
        """
        print(f"Processing video: {video_path}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or fps_default
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        print(f"Loaded {len(frames)} frames at {fps:.1f} FPS")
        
        # Process frames
        clip_features = []
        prev_poses = []
        
        for i, frame in enumerate(frames):
            if i % 30 == 0:  # Progress indicator
                print(f"Processing frame {i}/{len(frames)}")
            
            features_out, prev_poses = self.process_frame(frame, prev_poses, fps)
            
            # Pad or crop to max_people
            while len(features_out) < self.max_people:
                features_out.append(np.zeros((17, 5), dtype=np.float32))
            features_out = features_out[:self.max_people]
            
            clip_features.append(features_out)
        
        # Convert to numpy array (T, P, V, F)
        clip_array = np.array(clip_features, dtype=np.float32)
        print(f"✅ Generated array shape: {clip_array.shape}")
        
        return clip_array
    
    def cleanup(self):
        """Clean up resources"""
        self.pose_estimator.close()
        print("✅ Cleaned up resources")


# Compatibility functions for existing pipeline
def detect_poses_robust(frame: np.ndarray, detector: RobustPoseDetector) -> List[np.ndarray]:
    """Compatibility function for existing pipeline"""
    return detector.detect_poses(frame)


def process_frame_robust(frame: np.ndarray, keypoints_with_scores: List[np.ndarray], 
                       prev_people: List[np.ndarray], fps: float, 
                       detector: RobustPoseDetector) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compatibility function for existing pipeline"""
    return detector.process_frame(frame, prev_people, fps)


def video_to_numpy_robust(video_path: str, detector: RobustPoseDetector, 
                         fps_default: float = 30.0) -> np.ndarray:
    """Compatibility function for existing pipeline"""
    return detector.video_to_numpy(video_path, fps_default)
