
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

class KFTrack:
    count = 0
    def __init__(self, bbox, keypoints):
        self.id = KFTrack.count; KFTrack.count += 1
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        # state: [x, y, w, h, vx, vy, vw, vh]
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i+4] = dt
        self.kf.H = np.zeros((4,8))
        self.kf.H[0,0] = self.kf.H[1,1] = self.kf.H[2,2] = self.kf.H[3,3] = 1.0
        self.kf.P *= 10.0
        self.kf.R *= 1.0
        self.kf.Q *= 0.01
        self.time_since_update = 0
        self.hits = 1
        self.keypoints = keypoints  # last kp
        x, y, w, h = self._xyxy_to_xywh(bbox)
        self.kf.x[:4] = np.array([[x],[y],[w],[h]])

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox, keypoints):
        x, y, w, h = self._xyxy_to_xywh(bbox)
        self.kf.update(np.array([x,y,w,h]))
        self.time_since_update = 0
        self.hits += 1
        self.keypoints = keypoints

    def get_bbox(self):
        x,y,w,h = self.kf.x[:4].flatten()
        return self._xywh_to_xyxy([x,y,w,h])

    @staticmethod
    def _xyxy_to_xywh(b):
        x = (b[0]+b[2])/2.0
        y = (b[1]+b[3])/2.0
        w = (b[2]-b[0])
        h = (b[3]-b[1])
        return x,y,w,h

    @staticmethod
    def _xywh_to_xyxy(b):
        x,y,w,h = b
        return np.array([x-w/2, y-h/2, x+w/2, y+h/2])

class Tracker:
    def __init__(self, max_age=30, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []

    def step(self, detections):
        # detections: list of (bbox, score, cls, kps(17x3))
        # Predict existing
        for t in self.tracks:
            t.predict()

        if len(detections)==0 and len(self.tracks)==0:
            return []

        # Match
        cost = np.ones((len(self.tracks), len(detections))) * 1.0
        for i,t in enumerate(self.tracks):
            tb = t.get_bbox()
            for j,d in enumerate(detections):
                db = d[0]
                cost[i,j] = 1 - iou(tb, db)  # Cost = 1 - IoU (lower is better)
        if len(self.tracks)>0 and len(detections)>0:
            r,c = linear_sum_assignment(cost)
            assigned = set()
            used_tracks = set()
            for i,j in zip(r,c):
                if cost[i,j] <= (1 - self.iou_threshold):  # Fixed: IoU >= threshold
                    self.tracks[i].update(detections[j][0], detections[j][3])
                    assigned.add(j); used_tracks.add(i)
            # new tracks for unassigned detections
            for j,d in enumerate(detections):
                if j not in assigned:
                    self.tracks.append(KFTrack(d[0], d[3]))
            # prune old
            survivors = []
            for idx,t in enumerate(self.tracks):
                if t.time_since_update <= self.max_age:
                    survivors.append(t)
            self.tracks = survivors
        else:
            # if no tracks or no detections: spawn new from detections
            for d in detections:
                self.tracks.append(KFTrack(d[0], d[3]))

        # Output confirmed tracks
        outputs = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                outputs.append((t.id, t.get_bbox(), t.keypoints))
        return outputs
