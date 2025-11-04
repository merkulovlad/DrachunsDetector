
import os, argparse, time, cv2, torch
import numpy as np
from step1_input import FrameSource
from step2_pose import PoseEstimator
from tracking import Tracker
from features import aggregate_track_window, pool_across_tracks
from model import BiLSTMClassifier
from config import PoseConfig, TrackConfig, FeatureConfig, RuntimeConfig

def load_classifier(ckpt_path, device):
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        return None, None
    state = torch.load(ckpt_path, map_location=device)
    model = BiLSTMClassifier(input_dim=state["Fdim"], hidden_size=state["hidden"],
                             num_layers=state["layers"], dropout=state["dropout"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["Fdim"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=0, help="webcam index or video file")
    ap.add_argument("--weights", default="yolov8n-pose.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq_len", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--clf", default=None, help="path to trained classifier ckpt")
    ap.add_argument("--alert_threshold", type=float, default=0.8)
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    pose_cfg = PoseConfig(weights=args.weights, device=args.device)
    track_cfg = TrackConfig()
    feat_cfg = FeatureConfig(seq_len=args.seq_len, stride=args.stride)
    run_cfg = RuntimeConfig(alert_threshold=args.alert_threshold)

    # IO
    src = 0 if (str(args.source).isdigit() and len(str(args.source))<4) else args.source
    fs = FrameSource(src)

    pose = PoseEstimator(weights=pose_cfg.weights, imgsz=pose_cfg.imgsz,
                         conf=pose_cfg.conf, iou=pose_cfg.iou, device=pose_cfg.device, max_det=pose_cfg.max_det)
    tracker = Tracker(max_age=track_cfg.max_age, min_hits=track_cfg.min_hits, iou_threshold=track_cfg.iou_threshold)

    clf, fdim = load_classifier(args.clf, args.device)

    # sliding window buffers
    win_frames = []
    os.makedirs(run_cfg.out_dir, exist_ok=True)
    alert_on = False
    alert_start = None

    while True:
        ok, frame = fs.read()
        if not ok:
            break

        dets = pose.infer(frame)
        tracks = tracker.step(dets)

        # draw
        for tid, bbox, kp in tracks:
            x1,y1,x2,y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            for (x,y,conf) in kp:
                if conf>0.1:
                    cv2.circle(frame, (int(x),int(y)), 2, (255,0,0), -1)
            cv2.putText(frame, f"ID {tid}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # windowing
        win_frames.append([t for t in tracks])  # shallow copy
        if len(win_frames) >= feat_cfg.seq_len:
            # aggregate features over this window
            track_kps = {}
            for fr_tracks in win_frames:
                for tid, bbox, kp in fr_tracks:
                    track_kps.setdefault(tid, []).append(kp)
            track_feats = []
            for tid, seq in track_kps.items():
                if len(seq) >= max(2, feat_cfg.seq_len//2):
                    track_feats.append(aggregate_track_window(seq))
            pooled = pool_across_tracks(track_feats)
            X = torch.from_numpy(pooled).float().unsqueeze(0).unsqueeze(1).to(args.device)  # (1,1,F)
            if clf is not None:
                with torch.no_grad():
                    logits = clf(X)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                prob_violent = float(probs[1])
                cv2.putText(frame, f"Violence prob: {prob_violent:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255 if prob_violent>run_cfg.alert_threshold else 0), 2)
                if prob_violent > run_cfg.alert_threshold and not alert_on:
                    alert_on = True; alert_start = time.time()
                    # save a short clip image
                    ts = int(alert_start)
                    cv2.imwrite(os.path.join(run_cfg.out_dir, f"alert_{ts}.jpg"), frame)
                if prob_violent <= run_cfg.alert_threshold and alert_on:
                    alert_on = False
            # slide
            win_frames = win_frames[feat_cfg.stride:]

        if args.view or src==0:
            cv2.imshow("Violence Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    fs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
