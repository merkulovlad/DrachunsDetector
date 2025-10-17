import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # completely hides all GPUs from TF
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # optional, avoids warnings
import numpy as np
import torch
from torch_geometric.data import Data
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm



model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']


# --- Pose detection by movenet ---
def detect_poses(frame):
    input_image = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = movenet(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()  # shape: [1,6,56]
    return keypoints_with_scores[0]  # 6 people max


# --- Feature extraction per frame ---
def process_frame(frame, keypoints_with_scores, prev_people, fps, threshold=0.4):
    h, w, _ = frame.shape
    new_people = []
    features_out = []

    for person in keypoints_with_scores:
        scores = person[2::3]
        if np.sum(scores > threshold) < 5:
            continue

        # Reshape into (17, 3) = (y, x, conf)
        keypoints = np.array(person[:51]).reshape((17, 3))

        # Scale back to pixel coords
        keypoints[:, 0] *= h  # y
        keypoints[:, 1] *= w  # x

        new_people.append(keypoints)

        # Normalize coordinates to [-1, 1]
        y_norm = (keypoints[:, 0] / h) * 2 - 1
        x_norm = (keypoints[:, 1] / w) * 2 - 1
        conf = keypoints[:, 2]

        # Compute velocities
        person_velocities = np.zeros((17, 2))
        if prev_people:
            prev_keypoints = min(
                prev_people, key=lambda pk: np.linalg.norm(pk[:, :2] - keypoints[:, :2])
            )
            dt = 1.0 / fps
            for j, ((y, x, c), (py, px, pc)) in enumerate(zip(keypoints, prev_keypoints)):
                if c > threshold and pc > threshold:
                    vx, vy = (x - px) / dt, (y - py) / dt
                    # Normalize velocity
                    vx /= w
                    vy /= h
                    person_velocities[j] = [vx, vy]

        # Final feature vector per joint
        person_features = np.stack([x_norm, y_norm, conf,
                                    person_velocities[:, 0],
                                    person_velocities[:, 1]], axis=-1)  # (17, 5)

        features_out.append(person_features)

    return features_out, new_people

def video_to_numpy(video_path, max_people=6, threshold=0.4, fps_default=30):
    clip_features = []
    prev_people = []

    if isinstance(video_path, str):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    else:
        frames = video_path
        fps = fps_default

    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints_with_scores = detect_poses(rgb_frame)

        features_out, prev_people = process_frame(frame, keypoints_with_scores, prev_people, fps, threshold)

        # Pad or crop people to max_people
        while len(features_out) < max_people:
            features_out.append(np.zeros((17, 5), dtype=np.float32))
        features_out = features_out[:max_people]

        clip_features.append(features_out)

    # (T, P, V, F)
    clip_array = np.array(clip_features, dtype=np.float32)
    return clip_array


COCO_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12)
]

def frame_to_graph(frame, edges=COCO_EDGES):
    """
    frame: (P, V, F) tensor (people, joints, features)
    returns: list of Data objects (one graph per person, empty ones skipped)
    """
    people_graphs = []
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    for person in frame:  # (V, F)
        # Skip empty person (all features == 0)
        if torch.all(person == 0):
            continue

        x = person  # node features (17, F)
        data = Data(x=x, edge_index=edge_index)
        people_graphs.append(data)

    return people_graphs


def clip_to_graphs(clip):
    """
    clip: (F, T, V, P)
    returns: list of [graphs per frame]
             len = T, each element = list of Data objects
    """
    graphs_per_frame = []
    F, T, V, P = clip.shape

    for t in range(T):
        frame = clip[:, t, :, :]        # (F, V, P)
        frame = frame.permute(2, 1, 0)  # -> (P, V, F)
        graphs = frame_to_graph(frame)
        graphs_per_frame.append(graphs)

    return graphs_per_frame


def predict_clip(model, npy_path):
    if isinstance(npy_path, str):
        clip = np.load(npy_path)
    else:
        clip = npy_path
    clip = torch.tensor(clip, dtype=torch.float32).permute(3, 0, 2, 1)
    graph_seq = clip_to_graphs(clip)
    with torch.no_grad():
        output = model([graph_seq])
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
    return pred, prob.cpu().numpy()
