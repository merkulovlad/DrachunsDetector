import argparse
import itertools
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import TrainConfig


class RWF2000Skeletons(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        key: str = "skeletons",
        channels: tuple[int, ...] | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.key = key
        self.channels = channels

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        self.samples: list[tuple[Path, int, int]] = []
        for label_name, label in [("Fight", 1), ("NonFight", 0)]:
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            for npz_path in sorted(label_dir.glob("*.npz")):
                with np.load(npz_path) as data:
                    n = data[self.key].shape[0]
                for idx in range(n):
                    self.samples.append((npz_path, label, idx))
        if not self.samples:
            raise RuntimeError(f"No skeleton files found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        npz_path, label, window_idx = self.samples[index]
        with np.load(npz_path) as data:
            window = data[self.key][window_idx]
            label_arr = data.get("labels")
            if label_arr is not None:
                label = int(label_arr[window_idx])

        target_len = window.shape[0]
        if self.split == "train":
            if window.shape[0] > 8 and np.random.rand() < 0.5:
                keep = np.sort(
                    np.random.choice(
                        window.shape[0],
                        int(window.shape[0] * np.random.uniform(0.8, 0.9)),
                        replace=False,
                    )
                )
                window = window[keep]
            if np.random.rand() < 0.7:
                noise = np.random.normal(0, 0.01, size=window[..., :2].shape)
                window[..., :2] += noise
            curr_len = window.shape[0]
            if curr_len < target_len:
                pad = np.zeros((target_len - curr_len, window.shape[1], window.shape[2]), dtype=window.dtype)
                window = np.concatenate([window, pad], axis=0)
            elif curr_len > target_len:
                window = window[:target_len]

        window_sel = window if self.channels is None else window[..., list(self.channels)]
        window_sel = np.transpose(window_sel, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(window_sel), torch.tensor(label, dtype=torch.long)


def build_coco_adjacency(num_joints: int = 17):
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 12),
        (5, 11),
        (6, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    A = torch.eye(num_joints)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    D = torch.diag(1.0 / torch.clamp(A.sum(dim=1), min=1.0))
    return D @ A


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, dropout=0.0):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = torch.einsum("nctv,vw->nctw", x, self.A)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x + res


class STGCN(nn.Module):
    def __init__(self, num_classes=2, in_channels=2, graph_nodes=17, base_channels=64, dropout=0.5):
        super().__init__()
        A = build_coco_adjacency(graph_nodes)
        self.register_buffer("A", A)
        channels = [base_channels, base_channels, base_channels, 2 * base_channels, 2 * base_channels, 4 * base_channels]
        strides = [1, 1, 1, 2, 1, 2]
        layers = []
        c_in = in_channels
        for c_out, s in zip(channels, strides):
            layers.append(GraphConv(c_in, c_out, self.A, stride=s, residual=True, dropout=dropout))
            c_in = c_out
        self.stgcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stgcn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_eval_once(
    hparams: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    weight_decay: float,
    max_epochs: int,
    patience: int,
):
    model = STGCN(
        num_classes=2,
        in_channels=7,
        graph_nodes=17,
        base_channels=hparams["base_channels"],
        dropout=hparams["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optim = torch.optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=1, threshold=1e-3)

    best = {"acc": -1.0, "f1": -1.0, "auc": -1.0, "loss": float("inf"), "state": None}
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for X, y in tqdm(train_loader, desc=f"Train ep{epoch+1}", leave=False):
            X = X.to(device)
            y = y.to(device)
            if hparams["joint_dropout"] > 0:
                mask = (torch.rand_like(X[:, :1]) > hparams["joint_dropout"]).float()
                X = X * mask
            logits = model(X)
            loss = criterion(logits, y)
            optim.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            bs = y.size(0)
            preds = logits.argmax(dim=1)
            loss_sum += float(loss.item()) * bs
            total += bs
            correct += int((preds == y).sum().item())
        train_acc = correct / max(1, total)

        model.eval()
        v_total, v_correct, v_loss_sum = 0, 0, 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                bs = y.size(0)
                preds = logits.argmax(dim=1)
                probs = logits.softmax(dim=1)[:, 1]
                v_loss_sum += float(loss.item()) * bs
                v_total += bs
                v_correct += int((preds == y).sum().item())
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
                all_probs.append(probs.cpu())

        v_loss = v_loss_sum / max(1, v_total)
        v_acc = v_correct / max(1, v_total)
        val_preds = torch.cat(all_preds).numpy()
        val_labels = torch.cat(all_labels).numpy()
        val_probs = torch.cat(all_probs).numpy()
        v_f1 = f1_score(val_labels, val_preds)
        v_auc = roc_auc_score(val_labels, val_probs)

        scheduler.step(v_acc)
        if v_acc > best["acc"] + 1e-4:
            best.update({"acc": v_acc, "f1": v_f1, "auc": v_auc, "loss": v_loss})
            best["state"] = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return best


def main():
    parser = argparse.ArgumentParser(description="Grid search for ST-GCN.")
    parser.add_argument("--root", default="precomputed_skeletons", help="Dataset root.")
    parser.add_argument("--save-best", default="checkpoints/grid_best.pth", help="Path to save best checkpoint.")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    cfg = TrainConfig()
    weight_decay = getattr(cfg, "weight_decay", 1e-4)
    device = torch.device(cfg.model_device if torch.cuda.is_available() or cfg.model_device != "cuda" else "cpu")

    train_ds = RWF2000Skeletons(args.root, split="train")
    val_ds = RWF2000Skeletons(args.root, split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)

    search_space = {
        "lr": [2e-4, 3e-4],
        "base_channels": [32, 64],
        "dropout": [0.3, 0.5],
        "joint_dropout": [0.0, 0.1],
    }

    combos = list(itertools.product(*(search_space[k] for k in search_space)))
    results = []
    best_overall = {"acc": -1.0}

    print(f"Running {len(combos)} configs...")
    for combo in combos:
        hparams = {k: v for k, v in zip(search_space.keys(), combo)}
        print(f"\n=== Config {hparams} ===")
        res = train_eval_once(
            hparams,
            train_loader,
            val_loader,
            device,
            weight_decay=weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        results.append({"res": res, "hparams": hparams})
        print(f"Best acc {res['acc']:.4f} | F1 {res['f1']:.4f} | AUC {res['auc']:.4f}")
        if res["acc"] > best_overall.get("acc", -1):
            best_overall = res
            best_overall["hparams"] = hparams

    print("\nGrid search summary (sorted by acc):")
    for item in sorted(results, key=lambda r: r["res"]["acc"], reverse=True):
        r = item["res"]
        hp = item["hparams"]
        print(f"acc={r['acc']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f} params={hp}")

    if best_overall.get("state") is not None:
        out_path = Path(args.save_best)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": best_overall["state"],
                "num_classes": cfg.num_classes,
                "in_channels": 7,
                "base_channels": best_overall["hparams"]["base_channels"],
                "dropout": best_overall["hparams"]["dropout"],
            },
            out_path,
        )
        print(f"\nSaved best checkpoint to {out_path}")
    print("\nBest overall:", best_overall)


if __name__ == "__main__":
    main()
