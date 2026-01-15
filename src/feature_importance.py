# src/feature_importance.py
# Permutation Feature Importance for a *selected trained Hybrid* SRP dynamogram model.
# Measures drop in Macro-F1 on TEST set after permuting each engineered feature.
#
# Run from project ROOT (example):
#   python src/feature_importance.py --csv_path data/final.csv --run_dir results/hybrid7_final_v1 --n_points 512
#
# Outputs:
#   results/<run_dir>/analysis_feature_importance/feature_importance.csv
#   results/<run_dir>/analysis_feature_importance/feature_importance.png

import os
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from preprocessing import resample_curve, normalize_xy
from models.hybrid import HybridModel
from features import (
    compute_features_7,
    compute_features_17,
    FEATURE_NAMES_7,
    FEATURE_NAMES_17,
)


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_checkpoint(run_dir: str) -> str:
    # Prefer best_* naming; otherwise take first .pt
    candidates = [
        os.path.join(run_dir, "best_hybrid7.pt"),
        os.path.join(run_dir, "best_hybrid17.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    pts = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(f"No checkpoint found in: {run_dir}")
    return pts[0]


def infer_variant_from_ckpt(ckpt_path: str) -> str:
    b = os.path.basename(ckpt_path).lower()
    if "hybrid17" in b:
        return "hybrid17"
    if "hybrid7" in b:
        return "hybrid7"
    # Fallback: check ckpt metadata
    return "hybrid7"


def load_scaler(run_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(run_dir, "feature_scaler.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Hybrid training must have produced feature_scaler.npz."
        )
    z = np.load(path)
    return z["mean"], z["std"]


def standardize(F: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd2 = sd.copy()
    sd2[sd2 < 1e-8] = 1.0
    return (F - mu) / sd2


# -----------------------------
# Data build (graph-level)
# -----------------------------
def build_hybrid_arrays_from_csv(
    csv_path: str,
    n_points: int,
    class_names: List[str],
    model_variant: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Builds graph-level arrays aligned to checkpoint class order.

    Returns:
      graph_ids: list[str]
      X_all: [N, 2, n_points]
      F_all: [N, n_feat]
      y_enc: [N] integer labels aligned to class_names
      feat_names: list[str]
    """
    df = pd.read_csv(csv_path)

    required = {"graph_id", "x", "y", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=["graph_id", "x", "y", "label"]).copy()
    df["graph_id"] = df["graph_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).copy()

    has_point_no = "point_no" in df.columns

    # Graph-level label (mode)
    label_by_graph = df.groupby("graph_id")["label"].agg(lambda s: s.mode().iloc[0]).to_dict()

    # Keep only graphs whose label exists in the checkpoint classes
    allowed = set(class_names)
    kept_graphs = [gid for gid, lab in label_by_graph.items() if lab in allowed]
    if not kept_graphs:
        raise RuntimeError(
            "After filtering by checkpoint classes, 0 graphs remain. "
            "Check that CSV 'label' values exactly match ckpt label_classes."
        )

    df = df[df["graph_id"].isin(kept_graphs)].copy()
    label_by_graph = df.groupby("graph_id")["label"].agg(lambda s: s.mode().iloc[0]).to_dict()

    graph_ids = sorted(label_by_graph.keys())
    labels_str = np.array([label_by_graph[gid] for gid in graph_ids], dtype=object)

    # Encode using checkpoint order
    label_to_idx = {lab: i for i, lab in enumerate(class_names)}
    y_enc = np.array([label_to_idx[lab] for lab in labels_str], dtype=np.int64)

    # Feature set selection
    if model_variant == "hybrid7":
        feat_fn = compute_features_7
        feat_names = list(FEATURE_NAMES_7)
    else:
        feat_fn = compute_features_17
        feat_names = list(FEATURE_NAMES_17)

    n_feat = len(feat_names)

    X_all = np.zeros((len(graph_ids), 2, n_points), dtype=np.float32)
    F_all = np.zeros((len(graph_ids), n_feat), dtype=np.float32)

    grouped = df.groupby("graph_id", sort=False)

    for i, gid in enumerate(graph_ids):
        g = grouped.get_group(gid)
        g = g.sort_values("point_no" if has_point_no else "x")

        x = g["x"].to_numpy(dtype=np.float64)
        y = g["y"].to_numpy(dtype=np.float64)

        x_res, y_res = resample_curve(x, y, n_points)
        x_res, y_res = normalize_xy(x_res, y_res)

        X_all[i, 0, :] = x_res.astype(np.float32)
        X_all[i, 1, :] = y_res.astype(np.float32)

        F_all[i, :] = feat_fn(x_res, y_res).astype(np.float32)

    return graph_ids, X_all, F_all, y_enc, feat_names


@torch.no_grad()
def predict_classes(model, X_sig: np.ndarray, F_feat: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    n = X_sig.shape[0]
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X_sig[i:i+batch_size]).to(device)
        fb = torch.from_numpy(F_feat[i:i+batch_size]).to(device)
        logits = model(xb, fb)
        p = torch.argmax(logits, dim=1).detach().cpu().numpy()
        preds.append(p)
    return np.concatenate(preds, axis=0)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True, help="CSV with columns: graph_id,x,y,label")
    ap.add_argument("--run_dir", required=True, help="Model run folder in results/ (contains best_hybrid*.pt)")
    ap.add_argument("--n_points", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    run_dir = args.run_dir
    ckpt_path = find_checkpoint(run_dir)
    ckpt = torch.load(ckpt_path, map_location=args.device)

    class_names = ckpt.get("label_classes", None)
    if not class_names:
        raise ValueError("Checkpoint missing 'label_classes'.")

    model_variant = infer_variant_from_ckpt(ckpt_path)

    # Use dropout from checkpoint if available
    dropout = float(ckpt.get("args", {}).get("dropout", 0.2))

    # Build graph-level arrays aligned to checkpoint class order
    graph_ids, X_all, F_all, y_all, feat_names = build_hybrid_arrays_from_csv(
        csv_path=args.csv_path,
        n_points=args.n_points,
        class_names=class_names,
        model_variant=model_variant
    )

    # Split: train/val/test (reproducible)
    X_trainval, X_test, F_trainval, F_test, y_trainval, y_test = train_test_split(
        X_all, F_all, y_all,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_all
    )
    val_relative = args.val_size / (1.0 - args.test_size)
    _X_train, _X_val, _F_train, _F_val, _y_train, _y_val = train_test_split(
        X_trainval, F_trainval, y_trainval,
        test_size=val_relative,
        random_state=args.seed,
        stratify=y_trainval
    )

    # Standardize features using training scaler saved in run_dir
    mu, sd = load_scaler(run_dir)
    if mu.shape[0] != F_test.shape[1]:
        raise RuntimeError(
            f"Scaler dim mismatch. scaler.mean has {mu.shape[0]} feats, but CSV features have {F_test.shape[1]}."
        )
    F_test_s = standardize(F_test, mu, sd).astype(np.float32)

    # Load model
    n_classes = len(class_names)
    n_feat = F_all.shape[1]

    model = HybridModel(n_classes=n_classes, n_feat=n_feat, dropout=dropout).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Baseline Macro-F1
    y_pred_base = predict_classes(model, X_test.astype(np.float32), F_test_s, args.device, args.batch_size)
    base_f1 = f1_score(y_test, y_pred_base, average="macro")
    print(f"Run: {run_dir}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Variant: {model_variant} | Classes: {n_classes} | Features: {n_feat}")
    print(f"Baseline Macro-F1 (test): {base_f1:.4f}")

    # Permutation importance
    rng = np.random.default_rng(args.seed)
    rows = []

    for j in range(n_feat):
        f1s = []
        for _ in range(args.n_repeats):
            Fp = F_test_s.copy()
            perm = rng.permutation(Fp.shape[0])
            Fp[:, j] = Fp[perm, j]

            y_pred = predict_classes(model, X_test.astype(np.float32), Fp, args.device, args.batch_size)
            f1 = f1_score(y_test, y_pred, average="macro")
            f1s.append(float(f1))

        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))
        drop = float(base_f1 - mean_f1)

        rows.append((feat_names[j], drop, mean_f1, std_f1))
        print(f"[{j+1:02d}/{n_feat}] {feat_names[j]:>22s} | drop={drop:.5f} | f1={mean_f1:.4f} ± {std_f1:.4f}")

    df = pd.DataFrame(rows, columns=["feature", "macro_f1_drop", "macro_f1_mean", "macro_f1_std"])
    df = df.sort_values("macro_f1_drop", ascending=False).reset_index(drop=True)

    # Save artifacts into dedicated analysis folder
    out_dir = os.path.join(run_dir, "analysis_feature_importance")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "feature_importance.csv")
    df.to_csv(out_csv, index=False)

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.barh(df["feature"][::-1], df["macro_f1_drop"][::-1])
    plt.title("Permutation Feature Importance (drop in Macro-F1)")
    plt.xlabel("Macro-F1 drop after permuting feature")
    plt.tight_layout()

    out_png = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(out_png, dpi=220)

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
