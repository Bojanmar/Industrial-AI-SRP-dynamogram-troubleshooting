# ============================================================
# SRP Dynamogram Diagnostic Demo (Streamlit)
# Hybrid CNN + 7 Engineered Features (hybrid7 only)
# ============================================================

import os
import sys
from typing import Tuple, Dict, Any, List

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
from features import compute_features_7
from models.hybrid import HybridModel


# ============================================================
# Streamlit page setup
# ============================================================
st.set_page_config(page_title="SRP Dynamogram Diagnostic Demo", layout="wide")

st.title("SRP Dynamogram Diagnostic Inference")
st.caption(
    "Industrial AI demo for SRP dynamogram diagnostics using a Hybrid CNN + 7 domain-engineered features."
)

with st.expander("ℹ️ What this demo shows / does not show", expanded=True):
    st.markdown(
        """
**This demo shows**
- End-to-end inference on SRP-like dynamogram data (tabular long format)
- Resampling + normalization for shape robustness
- Hybrid model: CNN shape encoder + 7 physics-informed engineered features
- **Top-2 predictions with confidence-aware decision support**

**This demo does NOT show**
- Proprietary field datasets or production weights
- Full-scale training runs
- Real-time deployment infrastructure

The goal is to demonstrate **methodology and decision logic**, not field deployment.
"""
    )


# ============================================================
# Data helpers
# ============================================================
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    required = {"graph_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. Required columns: graph_id, x, y"
        )

    df["graph_id"] = df["graph_id"].astype(str).str.strip()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["graph_id", "x", "y"]).copy()
    return df


def list_graph_ids(df: pd.DataFrame) -> List[str]:
    return sorted(df["graph_id"].unique().tolist())


def get_curve(df: pd.DataFrame, graph_id: str) -> Tuple[np.ndarray, np.ndarray]:
    g = df[df["graph_id"] == graph_id].copy()
    if g.empty:
        raise ValueError(f"No data found for graph_id = {graph_id}")

    # preserve ordering if available
    if "point_no" in g.columns:
        g = g.sort_values("point_no")
    else:
        g = g.sort_values("x")

    return g["x"].to_numpy(dtype=np.float64), g["y"].to_numpy(dtype=np.float64)


# ============================================================
# Model loading (Hybrid-7 only)
# ============================================================
@st.cache_resource
def load_model(run_dir: str, device_str: str):
    ckpt_path = os.path.join(run_dir, "best_hybrid7.pt")
    scaler_path = os.path.join(run_dir, "feature_scaler.npz")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Expected best_hybrid7.pt in: {run_dir}")

    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Expected feature_scaler.npz in: {run_dir}")

    # safe device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"

    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device)

    class_names = ckpt.get("label_classes", None)
    if class_names is None:
        raise ValueError("Checkpoint missing 'label_classes'.")

    class_names = [str(c).strip() for c in class_names]

    model = HybridModel(
        n_classes=len(class_names),
        n_feat=7,
        dropout=float(ckpt.get("args", {}).get("dropout", 0.2)),
    )

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    scaler = np.load(scaler_path)
    return model, class_names, scaler, ckpt_path, device, device_str


# ============================================================
# Inference
# ============================================================
def predict_one(
    model,
    scaler,
    class_names: List[str],
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_points: int = 512,
) -> Dict[str, Any]:
    # resample + normalize
    x_res, y_res = resample_curve(x, y, n_points=n_points)
    x_res, y_res = normalize_xy(x_res, y_res)

    # signal tensor: [1, 2, n_points]
    X = np.stack([x_res, y_res], axis=0).astype(np.float32)
    xb = torch.from_numpy(X).unsqueeze(0).to(device=device, dtype=torch.float32)

    # features: [1, 7] then standardize
    feat = compute_features_7(x_res, y_res).astype(np.float32).reshape(1, -1)

    mu = np.asarray(scaler["mean"]).reshape(1, -1)
    sd = np.asarray(scaler["std"]).reshape(1, -1)
    sd = np.where(sd < 1e-8, 1.0, sd)

    feat_s = (feat - mu) / sd
    fb = torch.from_numpy(feat_s).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        logits = model(xb, fb)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)

    order = np.argsort(-probs)
    i1, i2 = int(order[0]), int(order[1])

    return {
        "probs": probs,
        "pred1_label": class_names[i1],
        "pred1_prob": float(probs[i1]),
        "pred2_label": class_names[i2],
        "pred2_prob": float(probs[i2]),
        "margin": float(probs[i1] - probs[i2]),
    }


# ============================================================
# Visualization
# ============================================================
def plot_dynamogram(x: np.ndarray, y: np.ndarray, title: str = ""):
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=8, alpha=0.9)
    plt.plot(x, y, linewidth=1, alpha=0.8)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)


def prob_bar_top2(class_names: List[str], probs: np.ndarray):
    order = np.argsort(-probs)[:2]
    labels = [class_names[i] for i in order]
    vals = [float(probs[i]) for i in order]

    fig = plt.figure(figsize=(6, 2.2))
    plt.barh(labels[::-1], vals[::-1])
    plt.xlabel("Probability")
    plt.title("Top probabilities (Top-2)")
    plt.xlim(0, 1.0)
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# Sidebar – guided workflow
# ============================================================
with st.sidebar:
    st.header("Step 1 – Upload data")
    uploaded = st.file_uploader(
        "Upload CSV (graph_id, x, y)",
        type=["csv"],
        help="Use the demo dataset from the data/ folder or provide your own CSV following the same schema.",
    )

    st.header("Step 2 – Model (hybrid7)")
    run_dir = st.text_input(
        "Model folder (results/...)",
        value=os.path.join("results", "hybrid7_final_v1"),
        help="Must contain best_hybrid7.pt and feature_scaler.npz",
    )

    device_str = st.selectbox("Device", ["cpu", "cuda"], index=0)
    if device_str == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA is not available here. Falling back to CPU.")
        device_str = "cpu"

    n_points = st.number_input(
        "Resampling points",
        min_value=128,
        max_value=2048,
        value=512,
        step=64,
        help="The curve is resampled to a fixed number of points for stable inference.",
    )

    st.header("Step 3 – Output policy")
    threshold = st.slider(
        "Top-1 confidence threshold",
        0.50,
        0.95,
        0.70,
        0.01,
        help="If Top-1 confidence is below this threshold, the app still shows Top-2 but keeps the output style consistent.",
    )

    st.markdown("---")
    st.caption("Tip: start with the demo dataset in `data/` to verify the pipeline.")


# ============================================================
# Main app logic
# ============================================================
if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

# Load data
try:
    df = load_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

graph_ids = list_graph_ids(df)
if not graph_ids:
    st.error("No graph_id values found in the uploaded CSV.")
    st.stop()

# Load model
try:
    model, class_names, scaler, ckpt_path, device, device_str = load_model(
        run_dir, device_str
    )
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.caption(f"Loaded checkpoint: `{ckpt_path}`")
st.caption(
    f"Model: **Hybrid-7** | Classes: **{len(class_names)}** | Graphs in file: **{len(graph_ids)}**"
)

# Layout:
# Left column -> selector + model output + Top-2 probabilities bar chart
# Right column -> plot + export
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Select dynamogram")
    selected_id = st.selectbox("graph_id", graph_ids, index=0)
    st.caption("Single-graph inference (interactive demo).")

# Compute inference ONCE
try:
    x, y = get_curve(df, selected_id)
    pred = predict_one(
        model=model,
        scaler=scaler,
        class_names=class_names,
        x=x,
        y=y,
        device=device,
        n_points=int(n_points),
    )
except Exception as e:
    st.error(f"Inference failed for graph_id={selected_id}: {e}")
    st.stop()

l1, p1 = pred["pred1_label"], pred["pred1_prob"]
l2, p2 = pred["pred2_label"], pred["pred2_prob"]
margin = pred["margin"]

# Left column: Model output BELOW dropdown and ABOVE bar chart
with col_left:
    st.subheader("Model output")

    # Always green prediction style
    st.success(f"**Prediction:** {l1}  \nConfidence: **{p1:.2f}**")

    st.markdown("### Top probabilities")
    prob_bar_top2(class_names, pred["probs"])

# Right column: Plot + Export
with col_right:
    st.subheader("Dynamogram plot")
    plot_dynamogram(x, y, title=f"graph_id: {selected_id}")

    st.divider()
    st.subheader("Export prediction (selected graph)")

    out_row = {
        "graph_id": str(selected_id),
        "pred1_label": l1,
        "pred1_prob": p1,
        "pred2_label": (None if p1 >= threshold else l2),
        "pred2_prob": (None if p1 >= threshold else p2),
        "margin": (None if p1 >= threshold else margin),
        "threshold": float(threshold),
        "output_mode": ("top1_only" if p1 >= threshold else "top2_shown"),
        "model": "hybrid7",
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
