from help_functions import *
import threading
from collections import deque
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class STGCN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, num_classes=2, dropout=0.3):
        super(STGCN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, batch_graph_seqs):
        """
        batch_graph_seqs: list of [graph_seq] (one per video)
        """
        batch_embs = []

        for graph_seq in batch_graph_seqs:  # loop over videos
            frame_embs = []

            for graphs in graph_seq:  # loop over frames
                if len(graphs) == 0:
                    frame_embs.append(torch.zeros(1, self.gcn1.out_channels, device=self.fc.weight.device))
                    continue

                person_embs = []
                for g in graphs:
                    x = g.x.to(self.fc.weight.device)
                    edge_index = g.edge_index.to(self.fc.weight.device)
                    h = F.relu(self.gcn1(x, edge_index))
                    h = F.relu(self.gcn2(h, edge_index))
                    h = global_mean_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
                    person_embs.append(h)

                frame_emb = torch.mean(torch.stack(person_embs), dim=0)
                frame_embs.append(frame_emb)

            frame_embs = torch.cat(frame_embs, dim=0).unsqueeze(0)  # (1, T, C)
            frame_embs = self.layer_norm(frame_embs)
            _, (h_n, _) = self.lstm(frame_embs)
            video_emb = h_n[-1]
            batch_embs.append(video_emb)

        batch_embs = torch.cat(batch_embs, dim=0)
        out = self.fc(self.dropout(batch_embs))
        return out


WINDOW_SEC = 1
FPS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STGCN(num_classes=2).to(device)
model.load_state_dict(torch.load("best_stgcn_model.pt", map_location=device))
model.eval()
print(device)


def run_realtime_video(source=0):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    print(f"[INFO] Source FPS: {fps}")

    frame_buffer = deque(maxlen=int(WINDOW_SEC * fps))
    prob_history = deque(maxlen=120)  # keep last ~2 minutes
    start_time = time.time()

    # Live plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 3))
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Violence probability")
    ax.grid(True)

    last_pred_time = 0
    current_prob = 0.0
    pred_label = "Analyzing..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame.copy())

        # Predict every 1 second
        if time.time() - last_pred_time >= WINDOW_SEC and len(frame_buffer) == frame_buffer.maxlen:
            last_pred_time = time.time()

            # Convert buffer → numpy clip (video_to_numpy handles per-frame pose features)
            frames_np = np.array(frame_buffer)
            npy_clip = video_to_numpy(frames_np)

            pred, prob = predict_clip(model, npy_clip)
            current_prob = float(prob[0, 1])
            pred_label = "VIOLENT" if pred == 1 else "NON-VIOLENT"

            prob_history.append((time.time() - start_time, current_prob))

        # --- Visualization ---
        color = (0, 0, 255) if current_prob > 0.5 else (0, 255, 0)
        cv2.putText(frame, f"{pred_label} ({current_prob:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.imshow("Violence Detection", frame)

        # --- Update plot ---
        if len(prob_history) > 1:
            t, p = zip(*prob_history)
            line.set_xdata(t)
            line.set_ydata(p)
            ax.set_xlim(max(0, t[-1] - 60), t[-1])
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


run_realtime_video("data/violent/cam2/6.mp4")