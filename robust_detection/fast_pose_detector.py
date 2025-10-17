"""
FAST Pose Detection Pipeline for M4 Apple Silicon
Optimized for speed while maintaining compatibility with existing STGCN pipeline
Target: 15-20 FPS processing speed
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List, Tuple, Optional
import time
import warnings
warnings.filterwarnings("ignore")

class FastPoseDetector:
    """
    Fast pose detection optimized for M4 Apple Silicon
    Uses MoveNet with smart optimizations for speed
    """
    
    def __init__(self, 
                 model_url: str = "https://tfhub.dev/google/movenet/multipose/lightning/1",
                 max_people: int = 6,
                 confidence_threshold: float = 0.3,
                 frame_skip: int = 1):
        """
        Args:
            model_url: MoveNet model URL
            max_people: Maximum people to detect (MoveNet limit)
            confidence_threshold: Minimum confidence for poses
            frame_skip: Process every Nth frame (1 = all frames)
        """
        self.max_people = max_people
        self.confidence_threshold = confidence_threshold
        self.frame_skip = frame_skip
        
        # Load MoveNet model
        print(f"Loading MoveNet model...")
        self.model = hub.load(model_url)
        self.movenet = self.model.signatures['serving_default']
        
        # Pre-allocate tensors for speed
        self.input_tensor = tf.zeros((1, 256, 256, 3), dtype=tf.int32)
        
        print("✅ Fast Pose Detector initialized")
    
    def detect_poses_fast(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Fast pose detection using MoveNet with optimizations
        """
        # Resize frame efficiently
        h, w = frame.shape[:2]
        
        # Skip if frame is too small
        if w < 64 or h < 64:
            return []
        
        # Resize to 256x256 for MoveNet (most efficient size)
        if w != 256 or h != 256:
            frame_resized = cv2.resize(frame, (256, 256))
        else:
            frame_resized = frame
        
        # Convert to tensor efficiently
        input_image = tf.expand_dims(frame_resized, axis=0)
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Run detection
        outputs = self.movenet(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()[0]  # shape: [6, 56]
        
        poses = []
        for person in keypoints_with_scores:
            scores = person[2::3]
            if np.sum(scores > self.confidence_threshold) < 3:  # Lower threshold for speed
                continue
            
            # Reshape into (17, 3) = (y, x, conf)
            keypoints = np.array(person[:51]).reshape((17, 3))
            
            # Scale back to original frame coordinates
            keypoints[:, 0] *= h  # y
            keypoints[:, 1] *= w  # x
            
            poses.append(keypoints)
        
        return poses
    
    def process_frame_fast(self, frame: np.ndarray, prev_poses: List[np.ndarray], fps: float = 30.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Fast frame processing with minimal computations
        """
        h, w = frame.shape[:2]
        
        # Detect poses
        current_poses = self.detect_poses_fast(frame)
        
        # Extract features efficiently
        features_out = []
        
        for pose in current_poses:
            # Normalize coordinates to [-1, 1]
            x_norm = (pose[:, 1] / w) * 2 - 1  # x coordinate
            y_norm = (pose[:, 0] / h) * 2 - 1  # y coordinate
            conf = pose[:, 2]
            
            # Compute velocities (simplified)
            velocities = np.zeros((17, 2))
            if prev_poses:
                # Find closest previous pose (simplified)
                best_prev = None
                best_distance = float('inf')
                
                for prev_pose in prev_poses:
                    # Use only confident joints for distance calculation
                    mask = (conf > 0.3) & (prev_pose[:, 2] > 0.3)
                    if np.sum(mask) < 3:
                        continue
                    
                    distances = np.sqrt((pose[mask, 0] - prev_pose[mask, 0])**2 + 
                                       (pose[mask, 1] - prev_pose[mask, 1])**2)
                    avg_distance = np.mean(distances)
                    
                    if avg_distance < best_distance:
                        best_distance = avg_distance
                        best_prev = prev_pose
                
                if best_prev is not None:
                    dt = 1.0 / fps
                    for j in range(17):
                        if conf[j] > 0.3 and best_prev[j, 2] > 0.3:
                            vx = (pose[j, 1] - best_prev[j, 1]) / dt
                            vy = (pose[j, 0] - best_prev[j, 0]) / dt
                            # Normalize velocity
                            velocities[j, 0] = vx / w
                            velocities[j, 1] = vy / h
            
            # Create feature vector (same format as original)
            person_features = np.stack([
                x_norm, y_norm, conf,
                velocities[:, 0], velocities[:, 1]
            ], axis=-1)  # (17, 5)
            
            features_out.append(person_features)
        
        return features_out, current_poses
    
    def video_to_numpy_fast(self, video_path: str, max_people: int = 6, fps_default: float = 30.0) -> np.ndarray:
        """
        Fast video processing with frame skipping and optimizations
        """
        print(f"Processing video: {video_path}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or fps_default
        
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame skipping for speed
            if frame_count % self.frame_skip == 0:
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        print(f"Loaded {len(frames)} frames at {fps:.1f} FPS (skipped {frame_count - len(frames)} frames)")
        
        # Process frames
        clip_features = []
        prev_poses = []
        
        start_time = time.time()
        
        for i, frame in enumerate(frames):
            if i % 30 == 0:  # Progress indicator
                elapsed = time.time() - start_time
                fps_current = i / elapsed if elapsed > 0 else 0
                print(f"Processing frame {i}/{len(frames)} (FPS: {fps_current:.1f})")
            
            features_out, prev_poses = self.process_frame_fast(frame, prev_poses, fps)
            
            # Pad or crop to max_people
            while len(features_out) < max_people:
                features_out.append(np.zeros((17, 5), dtype=np.float32))
            features_out = features_out[:max_people]
            
            clip_features.append(features_out)
        
        # Convert to numpy array (T, P, V, F)
        clip_array = np.array(clip_features, dtype=np.float32)
        
        total_time = time.time() - start_time
        processing_fps = len(frames) / total_time
        
        print(f"✅ Generated array shape: {clip_array.shape}")
        print(f"Processing time: {total_time:.2f} seconds")
        print(f"Processing FPS: {processing_fps:.1f}")
        
        return clip_array


# Ultra-fast version with aggressive optimizations
class UltraFastPoseDetector(FastPoseDetector):
    """
    Ultra-fast pose detection with aggressive optimizations
    Target: 20+ FPS processing speed
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_skip = 2  # Process every 2nd frame
        self.confidence_threshold = 0.2  # Lower threshold
        print("✅ Ultra-Fast Pose Detector initialized")
    
    def detect_poses_fast(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Ultra-fast pose detection with aggressive optimizations
        """
        # Skip very small frames
        h, w = frame.shape[:2]
        if w < 128 or h < 128:
            return []
        
        # Use smaller input size for speed (192x192 instead of 256x256)
        frame_resized = cv2.resize(frame, (192, 192))
        
        # Convert to tensor
        input_image = tf.expand_dims(frame_resized, axis=0)
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Run detection
        outputs = self.movenet(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()[0]
        
        poses = []
        for person in keypoints_with_scores:
            scores = person[2::3]
            if np.sum(scores > self.confidence_threshold) < 2:  # Very low threshold
                continue
            
            # Reshape into (17, 3)
            keypoints = np.array(person[:51]).reshape((17, 3))
            
            # Scale back to original frame coordinates
            keypoints[:, 0] *= h
            keypoints[:, 1] *= w
            
            poses.append(keypoints)
        
        return poses


# Compatibility functions
def video_to_numpy_fast(video_path: str, detector: FastPoseDetector, 
                       max_people: int = 6, fps_default: float = 30.0) -> np.ndarray:
    """Compatibility function for existing pipeline"""
    return detector.video_to_numpy_fast(video_path, max_people, fps_default)
