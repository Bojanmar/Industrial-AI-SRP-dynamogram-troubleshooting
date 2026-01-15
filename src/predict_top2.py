# src/predict_top2.py
# Run inference with trained model and export Top-2 predictions with confidences.
#
# Works for:
#   --model cnn
#   --model hybrid7
#   --model hybrid17
#
# Output CSV columns:
# graph_id,true_label,pred1_label,pred1_prob,pred2_label,pred2_prob,margin
#
# margin = pred1_prob - pred2_prob  (small margin => ambiguous case)

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from preprocessing import load_and_prepare_signal
from features import (
    compute_features_7, compute_features_17,
    standardize_features, FEATURE_NAMES_7, FEATURE_NAMES_17
)
from models.cnn1d import Dyn1DCNN
from models.hybrid import HybridModel


class DynDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.from_numpy(X)
        self.y = None if y is None else torch.from_numpy(y)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class HybridDynDataset(Dataset):
    def __init__(self, X: np.ndarray, F: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.from_numpy(X)
        self.F = torch.from_numpy(F)
        self.y = None if y is None else torch.from_numpy(y)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx], self.F[idx]
        return self.X[idx], self.F[idx], self.y[idx]


def build_feature_matrix(X_all: np.ndarray, feature_mode: str) -> tuple[np.ndarray, list[str]]:
    N = X_all.shape[0]
    if feature_mode == "hybrid7":
        F = np.zeros((N, 7), dtype=np.float32)
        for i in range(N):
            F[i, :] = compute_features_7(X_all[i, 0, :], X_all[i, 1, :])
        return F, FEATURE_NAMES_7

    if feature_mode == "hybrid17":
        F = np.zeros((N, 17), dtype=np.float32)
        for i in range(N):
            F[i, :] = compute_features_17(X_all[i, 0, :], X_all[i, 1, :])
        return F, FEATURE_NAMES_17

    raise ValueError("feature_mode must be 'hybrid7' or 'hybrid17'")


@torch.no_grad()
def predict_topk_cnn(model, loader, device, k=2):
    model.eval()
    all_topk_idx = []
    all_topk_prob = []
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        topk_prob, topk_idx = torch.topk(probs, k=k, dim=1)
        all_topk_idx.append(topk_idx.cpu().numpy())
        all_topk_prob.append(topk_prob.cpu().numpy())
    return np.vstack(all_topk_idx), np.vstack(all_topk_prob)


@torch.no_grad()
def predict_topk_hybrid(model, loader, device, k=2):
    model.eval()
    all_topk_idx = []
    all_topk_prob = []
    for xb, fb in loader:
        xb = xb.to(device)
        fb = fb.to(device)
        logits = model(xb, fb)
        probs = torch.softmax(logits, dim=1)
        topk_prob, topk_idx = torch.topk(probs, k=k, dim=1)
        all_topk_idx.append(topk_idx.cpu().numpy())
        all_topk_prob.append(topk_prob.cpu().numpy())
    return np.vstack(all_topk_idx), np.vstack(all_topk_prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True, help="CSV with columns: graph_id,x,y,label,(optional point_no)")
    ap.add_argument("--model", choices=["cnn", "hybrid7", "hybrid17"], default="hybrid17")
    ap.add_argument("--checkpoint", required=True, help="Path to best_*.pt checkpoint")
    ap.add_argument("--out_csv", default="results/top2_predictions.csv", help="Output CSV path")
    ap.add_argument("--n_points", type=int, default=512)
    ap.add_argument("--min_class_count", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--device", default=None, help="cuda/cpu (optional)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Load and prepare signals (same preprocessing as training)
    graph_ids, X_all, y_all, le, labels_str = load_and_prepare_signal(
        csv_path=args.csv_path,
        n_points=args.n_points,
        min_class_count=args.min_class_count,
        save_dir=out_dir,  # saves mapping
    )

    # Load checkpoint (for label order)
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt.get("label_classes", list(le.classes_))
    n_classes = len(class_names)

    if args.model == "cnn":
        model = Dyn1DCNN(n_classes=n_classes, dropout=0.0).to(device)
        model.load_state_dict(ckpt["model_state"])

        loader = DataLoader(
            DynDataset(X_all),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        topk_idx, topk_prob = predict_topk_cnn(model, loader, device, k=2)

    else:
        # Build feature matrix
        F_all, feat_names = build_feature_matrix(X_all, args.model)

        # Standardize features using scaler saved during training (if present)
        # If not found, fall back to standardizing on full data (less ideal but works).
        scaler_path = os.path.join(os.path.dirname(args.checkpoint), "feature_scaler.npz")
        if os.path.exists(scaler_path):
            sc = np.load(scaler_path)
            mu = sc["mean"]
            sd = sc["std"]
            sd = np.where(sd < 1e-8, 1.0, sd)
            F_all_s = (F_all - mu) / sd
        else:
            mu = F_all.mean(axis=0, keepdims=True)
            sd = F_all.std(axis=0, keepdims=True)
            sd[sd < 1e-8] = 1.0
            F_all_s = (F_all - mu) / sd

        model = HybridModel(n_classes=n_classes, n_feat=F_all_s.shape[1], dropout=0.0).to(device)
        model.load_state_dict(ckpt["model_state"])

        loader = DataLoader(
            HybridDynDataset(X_all, F_all_s),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        topk_idx, topk_prob = predict_topk_hybrid(model, loader, device, k=2)

    # Build output table
    pred1_idx = topk_idx[:, 0]
    pred2_idx = topk_idx[:, 1]
    pred1_prob = topk_prob[:, 0]
    pred2_prob = topk_prob[:, 1]
    margin = pred1_prob - pred2_prob

    df_out = pd.DataFrame({
        "graph_id": graph_ids,
        "true_label": labels_str,
        "pred1_label": [class_names[i] for i in pred1_idx],
        "pred1_prob": pred1_prob,
        "pred2_label": [class_names[i] for i in pred2_idx],
        "pred2_prob": pred2_prob,
        "margin": margin,
    })

    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved Top-2 predictions: {args.out_csv}")
    print("Tip: sort by 'margin' ascending to see most ambiguous cases first.")


if __name__ == "__main__":
    main()
