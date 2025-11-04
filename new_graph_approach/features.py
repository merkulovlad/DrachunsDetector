
import numpy as np

def kp_center(kp):
    # kp: (17,3) -> use visible joints only
    vis = kp[:,2] > 0.1
    if vis.sum() == 0:
        return np.array([0.0,0.0])
    return kp[vis,:2].mean(axis=0)

def pairwise_dists(a, b):
    return np.linalg.norm(a-b, axis=-1)

def angles_from_triplets(kp, triplets):
    # triplets: list of (i,j,k) -> angle at j (i-j-k)
    angs = []
    for i,j,k in triplets:
        if min(kp[i,2],kp[j,2],kp[k,2]) <= 0.1:
            angs.append(0.0); continue
        v1 = kp[i,:2]-kp[j,:2]
        v2 = kp[k,:2]-kp[j,:2]
        n1 = np.linalg.norm(v1)+1e-6
        n2 = np.linalg.norm(v2)+1e-6
        cos = np.clip((v1@v2)/(n1*n2), -1.0, 1.0)
        angs.append(np.arccos(cos))
    return np.array(angs, dtype=np.float32)

DEFAULT_TRIPLETS = [
    (5,7,9),  # left arm
    (6,8,10), # right arm
    (11,13,15), # left leg
    (12,14,16), # right leg
    (5,11,13), (6,12,14), # hip chains
    (5,6,12), (6,5,11) # shoulders/hips
]

def per_frame_features(kp_curr, kp_prev=None, kp_prev2=None):
    # Returns vector of features for a single person frame
    # Basic: center velocity/accel, mean joint speed, angle set, bbox size proxy
    center = kp_center(kp_curr)
    size_proxy = np.max(kp_curr[:,:2], axis=0) - np.min(kp_curr[:,:2], axis=0)
    size_proxy = np.linalg.norm(size_proxy)

    vel = np.zeros(2); acc = np.zeros(2)
    mean_speed = 0.0
    if kp_prev is not None:
        prev_center = kp_center(kp_prev)
        vel = center - prev_center
        mean_speed = np.mean(np.linalg.norm(kp_curr[:,:2]-kp_prev[:,:2], axis=1))
    if kp_prev2 is not None and kp_prev is not None:
        prev2_center = kp_center(kp_prev2)
        acc = center - 2*prev_center + prev2_center

    angs = angles_from_triplets(kp_curr, DEFAULT_TRIPLETS)
    feat = np.concatenate([center, vel, acc, [mean_speed, size_proxy], angs])
    return feat.astype(np.float32)

def aggregate_track_window(track_kp_seq):
    # track_kp_seq: list of kp arrays for a window
    # compute per-frame feats then aggregate (mean, max, std)
    pf = []
    for i,kp in enumerate(track_kp_seq):
        kp_prev = track_kp_seq[i-1] if i-1>=0 else None
        kp_prev2 = track_kp_seq[i-2] if i-2>=0 else None
        pf.append(per_frame_features(kp, kp_prev, kp_prev2))
    pf = np.stack(pf, axis=0)  # T x F
    stats = [pf.mean(axis=0), pf.max(axis=0), pf.std(axis=0)]
    return np.concatenate(stats, axis=0)  # 3F

def pool_across_tracks(list_of_track_features):
    # combine across multiple persons (max + mean)
    if len(list_of_track_features)==0:
        # Return consistent dummy feature dimension
        # per_frame_features: 16, aggregate_track_window: 3*16=48, pool_across_tracks: 2*48=96
        return np.zeros((96,), dtype=np.float32)
    X = np.stack(list_of_track_features, axis=0)
    return np.concatenate([X.mean(axis=0), X.max(axis=0)], axis=0)
