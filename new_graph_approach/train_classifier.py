import os, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import PoseConfig, TrackConfig, FeatureConfig, TrainConfig
from rwf2000_dataset import RWF2000Clips
from model import BiLSTMClassifier

def collate(batch):
    # batch: list of (X_windows, y_windows)
    Xs, ys = zip(*batch)
    # concat along windows to form a long list of clips
    X = torch.cat(Xs, dim=0)  # (Nclips, F)
    y = torch.cat(ys, dim=0)  # (Nclips,)
    # reshape to (B,T,F) for LSTM; we treat each clip as a single timestep sequence (T=1)
    # But to leverage temporal, we can simulate small T by chunking features if desired.
    # Simpler: expand dims to T=1
    X = X.unsqueeze(1)
    return X.float(), y.long()

def get_device(device_str):
    """Get the best available device, with MPS support for Mac"""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rwf2000", required=True, help="Path to RWF-2000 root (contains train/val/Fight/NonFight)")
    ap.add_argument("--weights", default="yolov8n-pose.pt")
    ap.add_argument("--epochs", type=int, default=None, help="Override max_epochs from config")
    ap.add_argument("--batch", type=int, default=None, help="Override batch_size from config")
    ap.add_argument("--lr", type=float, default=None, help="Override lr from config")
    ap.add_argument("--device", default=None, help="Override model_device from config (pose always uses CPU)")
    ap.add_argument("--workers", type=int, default=None, help="Override workers from config")
    ap.add_argument("--seq_len", type=int, default=None, help="Override seq_len from config")
    ap.add_argument("--stride", type=int, default=None, help="Override stride from config")
    ap.add_argument("--hidden", type=int, default=None, help="Override hidden_size from config")
    ap.add_argument("--layers", type=int, default=None, help="Override num_layers from config")
    ap.add_argument("--dropout", type=float, default=None, help="Override dropout from config")
    args = ap.parse_args()

    # Load config from config.py
    train_cfg = TrainConfig()
    
    # Override with command line arguments if provided
    if args.epochs is not None:
        train_cfg.max_epochs = args.epochs
    if args.batch is not None:
        train_cfg.batch_size = args.batch
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.device is not None:
        train_cfg.model_device = args.device
    if args.workers is not None:
        train_cfg.workers = args.workers
    if args.hidden is not None:
        train_cfg.hidden_size = args.hidden
    if args.layers is not None:
        train_cfg.num_layers = args.layers
    if args.dropout is not None:
        train_cfg.dropout = args.dropout

    # Device configuration - separate for pose vs model
    pose_device = get_device(train_cfg.pose_device)  # Always CPU for YOLO compatibility
    model_device = get_device(train_cfg.model_device)  # Auto-detect best for BiLSTM
    
    print(f"🔧 Device Configuration:")
    print(f"   YOLO-Pose: {pose_device} (for pose detection)")
    print(f"   BiLSTM Model: {model_device} (for training)")

    # Create configs using TrainConfig values
    pose_cfg = PoseConfig(weights=args.weights, device=pose_device)
    track_cfg = TrackConfig()
    feat_cfg = FeatureConfig(
        seq_len=args.seq_len if args.seq_len is not None else 30,
        stride=args.stride if args.stride is not None else 15
    )

    train_ds = RWF2000Clips(args.rwf2000, split="train", seq_len=feat_cfg.seq_len, stride=feat_cfg.stride,
                            pose_cfg=pose_cfg, track_cfg=track_cfg)
    val_ds = RWF2000Clips(args.rwf2000, split="val", seq_len=feat_cfg.seq_len, stride=feat_cfg.stride,
                          pose_cfg=pose_cfg, track_cfg=track_cfg)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, 
                              num_workers=train_cfg.workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, 
                           num_workers=train_cfg.workers, collate_fn=collate)

    # Determine feature dimension using a tiny probe
    X_probe, _ = next(iter(train_loader))
    Fdim = X_probe.shape[-1]

    model = BiLSTMClassifier(
        input_dim=Fdim, 
        hidden_size=train_cfg.hidden_size, 
        num_layers=train_cfg.num_layers, 
        dropout=train_cfg.dropout
    ).to(model_device)
    
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"🚀 Starting training with {train_cfg.max_epochs} epochs...")
    print(f"   Batch size: {train_cfg.batch_size}")
    print(f"   Learning rate: {train_cfg.lr}")
    print(f"   Model: {train_cfg.hidden_size} hidden, {train_cfg.num_layers} layers, {train_cfg.dropout} dropout")
    
    for epoch in range(train_cfg.max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.max_epochs}")
        total, correct, loss_sum = 0, 0, 0.0
        for X, y in pbar:
            X = X.to(model_device); y = y.to(model_device)
            logits = model(X)
            loss = criterion(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            loss_sum += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred==y).sum().item())
            total += int(y.size(0))
            pbar.set_postfix(loss=loss_sum/max(1,total), acc=correct/max(1,total))
        # val
        model.eval()
        v_total, v_correct = 0,0
        with torch.no_grad():
            for X,y in val_loader:
                X = X.to(model_device); y = y.to(model_device)
                logits = model(X)
                pred = logits.argmax(dim=1)
                v_correct += int((pred==y).sum().item())
                v_total += int(y.size(0))
        v_acc = v_correct/max(1,v_total)
        print(f"Val acc: {v_acc:.4f}")
        if v_acc > best_acc:
            best_acc = v_acc
            ckpt = f"checkpoints/bilstm_best.pt"
            torch.save({
                "model": model.state_dict(), 
                "Fdim": Fdim,
                "hidden": train_cfg.hidden_size, 
                "layers": train_cfg.num_layers, 
                "dropout": train_cfg.dropout
            }, ckpt)
            print("Saved", ckpt)

    print("Best val acc:", best_acc)

if __name__ == "__main__":
    main()