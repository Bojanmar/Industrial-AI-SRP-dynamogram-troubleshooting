# src/preprocessing.py
# Data format: multiple rows per graph_id
# Required columns: graph_id, x, y, label
# Optional: point_no

import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder

def resample_curve(x: np.ndarray, y: np.ndarray, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample (x, y) to fixed n_points using interpolation over a normalized x-axis.
    Steps:
      - sort by x
      - remove duplicate x (keep first)
      - normalize x to [0,1]
      - interpolate y to uniform grid
    """
    if len(x) < 2:
        x_new = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
        y_new = np.zeros(n_points, dtype=np.float32)
        return x_new, y_new

    idx = np.argsort(x)
    x = x[idx].astype(np.float64)
    y = y[idx].astype(np.float64)

    uniq_x, uniq_idx = np.unique(x, return_index=True)
    x = uniq_x
    y = y[uniq_idx]

    if len(x) < 2:
        x_new = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
        y_new = np.zeros(n_points, dtype=np.float32)
        return x_new, y_new

    x_min, x_max = float(x.min()), float(x.max())
    if math.isclose(x_max - x_min, 0.0):
        x_norm = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
    else:
        x_norm = (x - x_min) / (x_max - x_min)

    x_new = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    y_new = np.interp(x_new, x_norm, y).astype(np.float32)
    return x_new.astype(np.float32), y_new.astype(np.float32)

def normalize_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    x already [0,1] after resampling.
    y scaled by max(abs(y)) to keep shape robustly.
    """
    y = y.astype(np.float32)
    denom = float(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else 1.0
    y = y / denom
    return x.astype(np.float32), y.astype(np.float32)

def load_and_prepare_signal(
    csv_path: str,
    n_points: int,
    min_class_count: int,
    save_dir: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, LabelEncoder, np.ndarray]:
    """
    Build:
      graph_ids: [N]
      X_all: [N, 2, n_points]
      y_enc: [N]
      le: LabelEncoder
      labels_str: [N] original labels
    """
    df = pd.read_csv(csv_path)

    required = {"graph_id", "x", "y", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=["graph_id", "x", "y", "label"]).copy()
    has_point_no = "point_no" in df.columns

    # graph-level label: mode
    label_by_graph: Dict[str, str] = df.groupby("graph_id")["label"].agg(lambda s: s.mode().iloc[0]).to_dict()
    graph_labels = pd.Series(label_by_graph)
    label_counts = graph_labels.value_counts()

    allowed_labels = set(label_counts[label_counts >= min_class_count].index.tolist())
    allowed_graphs = [gid for gid, lab in label_by_graph.items() if lab in allowed_labels]
    df = df[df["graph_id"].isin(allowed_graphs)].copy()

    # refresh mapping after filtering
    label_by_graph = df.groupby("graph_id")["label"].agg(lambda s: s.mode().iloc[0]).to_dict()
    graph_ids = sorted(list(label_by_graph.keys()))
    labels_str = np.array([label_by_graph[gid] for gid in graph_ids], dtype=object)

    le = LabelEncoder()
    y_enc = le.fit_transform(labels_str).astype(np.int64)

    # Save label mapping
    mapping = pd.DataFrame({"class_index": np.arange(len(le.classes_)), "label": le.classes_})
    mapping_path = os.path.join(save_dir, "label_mapping_kept.csv")
    mapping.to_csv(mapping_path, index=False)

    X_all = np.zeros((len(graph_ids), 2, n_points), dtype=np.float32)

    grouped = df.groupby("graph_id", sort=False)
    for i, gid in enumerate(graph_ids):
        g = grouped.get_group(gid)
        if has_point_no:
            g = g.sort_values("point_no")
        else:
            g = g.sort_values("x")

        x = g["x"].to_numpy()
        y = g["y"].to_numpy()

        x_res, y_res = resample_curve(x, y, n_points)
        x_res, y_res = normalize_xy(x_res, y_res)

        X_all[i, 0, :] = x_res
        X_all[i, 1, :] = y_res

    return graph_ids, X_all, y_enc, le, labels_str
