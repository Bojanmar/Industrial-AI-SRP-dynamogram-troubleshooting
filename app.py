import os
import sys
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

# --- allow imports from src/ ---
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import resample_curve, normalize_xy
from features import compute_features_7, compute_features_17
from models.cnn1d import Dyn1DCNN
from models.hybrid import HybridModel


# -----------------------------
# Data helpers
# -----------------------------
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    required = {"graph_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Required: graph_id,x,y")

    # normalize dtypes
    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # drop invalid rows
    df = df.dropna(subset=["graph_id", "x", "y"]).copy()

    # optional cleanup: remove inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"]).copy()

    return df


def list_graph_ids(df: pd.DataFrame) -> List[str]:
    return sorted(df["graph_id"].unique().tolist())


def get_curve(df: pd.DataFrame, graph_id: str) -> Tuple[np.ndarray, np.ndarray]:
    gid = str(graph_id).strip()
    g = df[df["graph_id"] == gid].copy()

    if g.empty:
        raise ValueError(f"No rows found for graph_id={gid}. Check dtype/format in CSV.")

    if "point_no" in g.columns:
        g = g.sort_values("point_no")
    else:
        g = g.sort_values("x")

    x = g["x"].to_numpy(dtype=np.float64)
    y = g["y"].to_numpy(dtype=np.float64)
    return x, y


def get_selected_df(df: pd.DataFrame, graph_id: str) -> pd.DataFrame:
    gid = str(graph_id).strip()
    g = df[df["graph_id"] == gid].copy()
    if "point_no" in g.columns:
        g = g.sort_values("point_no")
    else:
        g = g.sort_values("x")
    return g


# -----------------------------
# Model helpers
# -----------------------------
def find_checkpoint(run_dir: str) -> Optional[str]:
    if not run_dir or not os.path.isdir(run_dir):
        return None

    # preferred names
    for fn in ["best_hybrid17.pt", "best_hybrid7.pt", "best_cnn.pt"]:
        p = os.path.join(run_dir, fn)
        if os.path.isfile(p):
            return p

    # fallback: any .pt
    pts = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith(".pt")]
    return pts[0] if pts else None


def infer_model_type(ckpt_path: str) -> str:
    base = os.path.basename(ckpt_path).lower()
    if "hybrid17" in base:
        return "hybrid17"
    if "hybrid7" in base:
        return "hybrid7"
    return "cnn"


def list_model_runs(results_root: str = "results") -> List[str]:
    if not os.path.isdir(results_root):
        return []
    runs = []
    for d in sorted(os.listdir(results_root)):
        p = os.path.join(results_root, d)
        if not os.path.isdir(p):
            continue
        ckpt = find_checkpoint(p)
        if ckpt is not None:
            runs.append(p)
    return runs


@st.cache_resource
def load_feature_scaler_cached(run_dir: str) -> Optional[Dict[str, np.ndarray]]:
    p = os.path.join(run_dir, "feature_scaler.npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p)
    mean = z["mean"]
    std = z["std"]
    return {"mean": mean, "std": std}


@st.cache_resource
def load_model(run_dir: str, device_str: str = "cpu"):
    ckpt_path = find_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            "Checkpoint not found in run_dir. Expected best_cnn.pt / best_hybrid7.pt / best_hybrid17.pt"
        )

    # safe device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"

    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device)

    class_names = ckpt.get("label_classes", None)
    if class_names is None:
        raise ValueError("Checkpoint missing 'label_classes'.")

    class_names = [str(c).strip() for c in class_names]

    model_type = infer_model_type(ckpt_path)

    if model_type == "cnn":
        model = Dyn1DCNN(
            n_classes=len(class_names),
            dropout=float(ckpt.get("args", {}).get("dropout", 0.2)),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return model, class_names, model_type, ckpt_path, device_str

    n_feat = ckpt.get("n_feat", None)
    if n_feat is None:
        n_feat = 17 if model_type == "hybrid17" else 7

    model = HybridModel(
        n_classes=len(class_names),
        n_feat=int(n_feat),
        dropout=float(ckpt.get("args", {}).get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, class_names, model_type, ckpt_path, device_str


def prepare_input_tensor(x: np.ndarray, y: np.ndarray, n_points: int = 512) -> np.ndarray:
    x_res, y_res = resample_curve(x, y, n_points=n_points)
    x_res, y_res = normalize_xy(x_res, y_res)
    X = np.stack([x_res, y_res], axis=0).astype(np.float32)  # [2, n_points]
    return X


def predict_one(
    model,
    class_names: List[str],
    model_type: str,
    x: np.ndarray,
    y: np.ndarray,
    run_dir: str,
    device_str: str = "cpu",
    n_points: int = 512,
) -> Dict[str, Any]:
    # safe device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    X = prepare_input_tensor(x, y, n_points=n_points)  # [2, n_points]
    xb = torch.from_numpy(X).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,2,n_points]

    with torch.no_grad():
        if model_type == "cnn":
            logits = model(xb)
        else:
            x_res = X[0, :]
            y_res = X[1, :]

            if model_type == "hybrid7":
                feat = compute_features_7(x_res, y_res).astype(np.float32).reshape(1, -1)
            else:
                feat = compute_features_17(x_res, y_res).astype(np.float32).reshape(1, -1)

            scaler = load_feature_scaler_cached(run_dir)
            if scaler is None:
                raise FileNotFoundError("Hybrid model requires feature_scaler.npz in run_dir.")

            mu = np.asarray(scaler["mean"]).reshape(1, -1)
            sd = np.asarray(scaler["std"]).reshape(1, -1)
            sd = np.where(sd < 1e-8, 1.0, sd)

            feat_s = (feat - mu) / sd
            fb = torch.from_numpy(feat_s).to(device=device, dtype=torch.float32)

            logits = model(xb, fb)

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)

    top_idx = np.argsort(-probs)[:2]
    i1, i2 = int(top_idx[0]), int(top_idx[1])

    return {
        "probs": probs,
        "pred1_label": class_names[i1],
        "pred1_prob": float(probs[i1]),
        "pred2_label": class_names[i2],
        "pred2_prob": float(probs[i2]),
        "margin": float(probs[i1] - probs[i2]),
    }


# -----------------------------
# Visualization
# -----------------------------
def plot_dynamogram(x: np.ndarray, y: np.ndarray, title: str = ""):
    fig = plt.figure(figsize=(8, 5))

    # ✅ Excel-like scatter
    plt.scatter(x, y, s=8, alpha=0.9)

    # ✅ linija preko tačaka (zatvara konturu vizuelno)
    plt.plot(x, y, linewidth=1, alpha=0.8)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)



def prob_bar(class_names: List[str], probs: np.ndarray, topk: int = 8):
    order = np.argsort(-probs)[:topk]
    labels = [class_names[i] for i in order]
    vals = [float(probs[i]) for i in order]

    fig = plt.figure()
    plt.barh(labels[::-1], vals[::-1])
    plt.xlabel("Probability")
    plt.title("Top probabilities")
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="SRP Dynamogram Inference", layout="wide")
st.title("SRP Dynamogram Inference (Production UI)")

with st.sidebar:
    st.header("1) Upload data")
    uploaded = st.file_uploader("Upload CSV (graph_id,x,y)", type=["csv"])

    st.header("2) Model")
    runs = list_model_runs("results")

    if runs:
        default_idx = 0
        for i, r in enumerate(runs):
            if r.replace("/", "\\").endswith("hybrid17_images_2026"):
                default_idx = i
                break

        run_dir = st.selectbox(
            "Select model run (results/...)",
            options=runs,
            index=default_idx,
            help="Choose trained model folder (must contain best_*.pt)"
        )
    else:
        run_dir = st.text_input(
            "Model run folder (results\\...)", 
            value=os.path.join("results", "hybrid17_images_2026"),
            help="Folder that contains best_*.pt and (for hybrid) feature_scaler.npz"
        )

    device_str = st.selectbox("Device", ["cpu", "cuda"], index=0)
    if device_str == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA not available on this machine. Falling back to CPU.")
        device_str = "cpu"

    n_points = st.number_input("Resample points (n_points)", min_value=128, max_value=2048, value=512, step=64)

    st.header("3) Output policy")
    threshold = st.slider("Top-1 confidence threshold", 0.50, 0.95, 0.70, 0.01)

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Load data
try:
    df = load_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

graph_ids = list_graph_ids(df)

# Load model
try:
    model, class_names, model_type, ckpt_path, device_str = load_model(run_dir, device_str=device_str)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.caption(f"Loaded checkpoint: {ckpt_path}")
st.caption(f"Model type: **{model_type}** | Classes: **{len(class_names)}** | Graphs in file: **{len(graph_ids)}**")

# Main UI: dropdown selection -> single inference
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Select dynamogram")
    selected_id = st.selectbox("graph_id", graph_ids, index=0)
    st.caption("Only the selected graph is passed through the model (single inference).")

    sel_df = get_selected_df(df, selected_id)
    st.divider()
    st.subheader("Selected graph: data (filtered)")
    st.caption(f"Rows for selected graph: {len(sel_df)}")
    st.dataframe(sel_df, use_container_width=True)

    with st.expander("Quick preview (first rows - full file)"):
        st.dataframe(df.head(20), use_container_width=True)

with col_right:
    x, y = get_curve(df, selected_id)

    st.subheader("Dynamogram plot")
    plot_dynamogram(x, y, title=f"graph_id: {selected_id}")

    st.subheader("Model prediction")
    pred = predict_one(
        model=model,
        class_names=class_names,
        model_type=model_type,
        x=x,
        y=y,
        run_dir=run_dir,
        device_str=device_str,
        n_points=int(n_points),
    )

    p1, p2, margin = pred["pred1_prob"], pred["pred2_prob"], pred["margin"]
    l1, l2 = pred["pred1_label"], pred["pred2_label"]

    if p1 >= threshold:
        st.success(f"**Predicted:** {l1}  \nConfidence: **{p1:.2f}**")
    else:
        st.warning(f"Top-1 confidence below {threshold:.2f} → showing Top-2.")
        st.write(f"**Top-1:** {l1} ({p1:.2f})")
        st.write(f"**Top-2:** {l2} ({p2:.2f})")
        st.caption(f"Margin (Top-1 − Top-2): {margin:.2f}")

    prob_bar(class_names, pred["probs"], topk=8)

    st.divider()
    st.subheader("Export prediction (selected graph)")
    out_row = {
        "graph_id": str(selected_id),
        "pred1_label": l1,
        "pred1_prob": p1,
        "pred2_label": (None if p1 >= threshold else l2),
        "pred2_prob": (None if p1 >= threshold else p2),
        "margin": (None if p1 >= threshold else margin),
        "threshold": threshold,
        "output_mode": ("top1_only" if p1 >= threshold else "top2_shown"),
        "model_type": model_type,
        "checkpoint": os.path.basename(ckpt_path),
        "device": device_str,
        "n_points": int(n_points),
    }
    out_df = pd.DataFrame([out_row])
    st.download_button(
        "Download prediction as CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_selected.csv",
        mime="text/csv",
    )
