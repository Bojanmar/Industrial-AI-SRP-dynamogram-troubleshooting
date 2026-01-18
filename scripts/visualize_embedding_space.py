# scripts/visualize_embedding_space.py
# ============================================================
# Embedding space visualization (UMAP + t-SNE)
# Extracts CNN-only embeddings from HybridModel.forward_embedding()
# and plots 2D projections.
#
# Input:  data/final.csv  (graph_id,x,y,label OR graph_id,x,y + labels in mapping)
# Model:  results/<run_dir>/best_hybrid7.pt (or best_hybrid17.pt)
# Output: results/embedding_space/<tag>_umap.png, <tag>_tsne.png, <tag>_embeddings.csv
# ============================================================

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# --- optional deps ---
try:
    import umap  
    UMAP_OK = True
except Exception:
    UMAP_OK = False

try:
    from sklearn.manifold import TSNE
    TSNE_OK = True
except Exception:
    TSNE_OK = False

# --- allow imports from src/ ---
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
SRC_DIR = os.path.join(REPO, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import resample_curve, normalize_xy
from models.hybrid import HybridModel


def load_curve(df: pd.DataFrame, gid: str):
    g = df[df["graph_id"] == gid].copy()
    if "point_no" in g.columns:
        g = g.sort_values("point_no")
    else:
        g = g.sort_values("x")
    x = g["x"].to_numpy(dtype=np.float64)
    y = g["y"].to_numpy(dtype=np.float64)
    return x, y


def safe_label_column(df: pd.DataFrame):
    # prefer 'label' if exists
    if "label" in df.columns:
        return "label"
    # sometimes 'y' or 'class' etc
    for c in ["y", "class", "target", "label_str"]:
        if c in df.columns:
            return c
    return None


def extract_embeddings(
    df: pd.DataFrame,
    model: HybridModel,
    n_points: int,
    device: torch.device,
):
    gids = sorted(df["graph_id"].astype(str).unique().tolist())
    Z = []
    for gid in gids:
        x, y = load_curve(df, gid)
        x_r, y_r = resample_curve(x, y, n_points=n_points)
        x_r, y_r = normalize_xy(x_r, y_r)

        X = np.stack([x_r, y_r], axis=0).astype(np.float32)  # [2, N]
        xb = torch.from_numpy(X).unsqueeze(0).to(device=device)  # [1,2,N]

        with torch.no_grad():
            z = model.forward_embedding(xb).detach().cpu().numpy().reshape(-1)  # [128]
        Z.append(z)

    Z = np.vstack(Z).astype(np.float32)
    return gids, Z


def plot_2d(points2d: np.ndarray, labels: np.ndarray | None, title: str, out_path: str):
    plt.figure(figsize=(10, 7))
    if labels is None:
        plt.scatter(points2d[:, 0], points2d[:, 1], s=6, alpha=0.85)
    else:
        # encode labels to ints for coloring
        uniq = pd.unique(labels)
        lut = {v: i for i, v in enumerate(uniq)}
        y = np.array([lut[v] for v in labels], dtype=int)
        plt.scatter(points2d[:, 0], points2d[:, 1], c=y, s=7, alpha=0.9)
        plt.colorbar(label="Label index")

    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="data/final.csv")
    ap.add_argument("--run_dir", default="results/hybrid7_final_v1")
    ap.add_argument("--ckpt_name", default="", help="Optional override (e.g. best_hybrid17.pt)")
    ap.add_argument("--n_points", type=int, default=512)
    ap.add_argument("--max_graphs", type=int, default=0, help="0 = all, else sample first N (for quick tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/embedding_space")
    ap.add_argument("--tag", default="", help="Output tag for filenames (default: run_dir name)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load data
    df = pd.read_csv(args.csv_path)
    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["graph_id", "x", "y"]).copy()

    label_col = safe_label_column(df)

    # optionally limit graphs
    if args.max_graphs and args.max_graphs > 0:
        gids = sorted(df["graph_id"].unique().tolist())
        gids = gids[: args.max_graphs]
        df = df[df["graph_id"].isin(gids)].copy()

    # ---- load checkpoint
    run_dir = Path(args.run_dir)
    if args.ckpt_name.strip():
        ckpt_path = run_dir / args.ckpt_name
    else:
        # try common names
        for cand in ["best_hybrid7.pt", "best_hybrid17.pt", "best.pt"]:
            if (run_dir / cand).is_file():
                ckpt_path = run_dir / cand
                break
        else:
            raise FileNotFoundError(f"No checkpoint found in {run_dir} (expected best_hybrid7.pt / best_hybrid17.pt / best.pt)")

    scaler_path = run_dir / "feature_scaler.npz"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing feature_scaler.npz in {run_dir}")

    device = torch.device("cpu")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    class_names = ckpt.get("label_classes", None)
    if class_names is None:
        raise ValueError("Checkpoint missing 'label_classes'.")

    scaler = np.load(str(scaler_path))
    n_feat = int(np.asarray(scaler["mean"]).ravel().size)

    model = HybridModel(n_classes=len(class_names), n_feat=n_feat, dropout=float(ckpt.get("args", {}).get("dropout", 0.2)))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # ---- embeddings
    gids, Z = extract_embeddings(df, model, args.n_points, device)

    tag = args.tag.strip() or run_dir.name
    emb_csv = out_dir / f"{tag}_embeddings.csv"

    emb_df = pd.DataFrame(Z, columns=[f"e{i}" for i in range(Z.shape[1])])
    emb_df.insert(0, "graph_id", gids)

    # attach labels if available (graph-level: take first label per graph)
    if label_col is not None:
        lbl_map = df.groupby("graph_id")[label_col].first()
        emb_df["label"] = emb_df["graph_id"].map(lbl_map).astype(str)

    emb_df.to_csv(emb_csv, index=False)
    print("Saved embeddings:", emb_csv)

    labels = emb_df["label"].values if "label" in emb_df.columns else None

    # ---- UMAP
    if UMAP_OK:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.seed)
        umap2d = reducer.fit_transform(Z)
        out_png = out_dir / f"{tag}_umap.png"
        plot_2d(umap2d, labels, f"UMAP — CNN embedding space ({tag})", str(out_png))
        print("Saved:", out_png)
    else:
        print("[WARN] umap-learn not installed. Skip UMAP. Install: pip install umap-learn")

    # ---- t-SNE
    if TSNE_OK:
        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30, init="pca", learning_rate="auto")
        tsne2d = tsne.fit_transform(Z)
        out_png = out_dir / f"{tag}_tsne.png"
        plot_2d(tsne2d, labels, f"t-SNE — CNN embedding space ({tag})", str(out_png))
        print("Saved:", out_png)
    else:
        print("[WARN] scikit-learn missing TSNE? (unlikely).")

    print("\nDone.")


if __name__ == "__main__":
    main()
