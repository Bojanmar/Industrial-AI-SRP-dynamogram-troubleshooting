# ============================================================
# SRP Dynamogram Diagnostic Demo (Streamlit)
# Decision regime:
#   1) Single-label hybrid7 (softmax) used when confident
#   2) If ambiguous (p1 < threshold), switch to multi-label hybrid7 (sigmoid)
# ============================================================

import os
import sys
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

import base64
import requests
from pathlib import Path
import hashlib


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
    "Industrial AI demo for SRP dynamogram diagnostics using Hybrid CNN + 7 domain-engineered features, "
    "with a production-style decision regime (single → multi on ambiguity)."
)

with st.expander("ℹ️ What this demo shows / does not show", expanded=True):
    st.markdown(
        """
**This demo shows**
- End-to-end inference on SRP-like dynamogram data (tabular long format)
- Resampling + normalization for shape robustness
- Hybrid model: CNN shape encoder + 7 physics-informed engineered features
- **Top-2 predictions with a production decision regime**
  - use **single-label** model when confident
  - fallback to **multi-label** model when ambiguous

**This demo does NOT show**
- Proprietary field datasets or production weights beyond the demo artifact
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


def short_file_hash(path: str, n: int = 8) -> str:
    """Return short SHA256 hash of a file (first n chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# ============================================================
# Private GitHub artifacts download
# ============================================================
def _github_download_file(repo_full: str, path_in_repo: str, token: str) -> bytes:
    """
    Download a file from a private GitHub repo via the Contents API.
    Returns raw bytes.
    """
    url = f"https://api.github.com/repos/{repo_full}/contents/{path_in_repo}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "streamlit-app",
    }
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(
            f"GitHub download failed for {repo_full}/{path_in_repo} "
            f"(status={r.status_code}): {r.text}"
        )

    data = r.json()
    if "content" not in data or "encoding" not in data:
        raise RuntimeError(f"Unexpected GitHub API response for {path_in_repo}: {data}")

    if data["encoding"] != "base64":
        raise RuntimeError(f"Unsupported encoding from GitHub API: {data['encoding']}")

    return base64.b64decode(data["content"])


def ensure_private_artifacts(local_run_dir: str) -> bool:
    """
    Ensure artifacts exist locally by downloading them from a private GitHub repo.
    Downloads:
      - best_hybrid7.pt (single-label)
      - best_hybrid7_multilabel.pt (multi-label)
      - feature_scaler.npz
    Returns True if something was downloaded, False if already cached.
    """
    token = st.secrets.get("GITHUB_TOKEN", None)
    repo = st.secrets.get("PRIVATE_REPO", None)

    single_pt_path = st.secrets.get("MODEL_SINGLE_PT_PATH", "best_hybrid7.pt")
    multi_pt_path = st.secrets.get("MODEL_MULTI_PT_PATH", "best_hybrid7_multilabel.pt")
    npz_path = st.secrets.get("SCALER_NPZ_PATH", "feature_scaler.npz")

    if not token or not repo:
        raise RuntimeError(
            "Missing Streamlit Secrets. Please set GITHUB_TOKEN and PRIVATE_REPO in Streamlit Secrets."
        )

    run_dir = Path(local_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    local_single = run_dir / "best_hybrid7.pt"
    local_multi = run_dir / "best_hybrid7_multilabel.pt"
    local_npz = run_dir / "feature_scaler.npz"

    downloaded = False

    if not local_single.exists():
        blob = _github_download_file(repo, single_pt_path, token)
        local_single.write_bytes(blob)
        downloaded = True

    if not local_multi.exists():
        blob = _github_download_file(repo, multi_pt_path, token)
        local_multi.write_bytes(blob)
        downloaded = True

    if not local_npz.exists():
        blob = _github_download_file(repo, npz_path, token)
        local_npz.write_bytes(blob)
        downloaded = True

    return downloaded


# ============================================================
# Model loading
# ============================================================
@st.cache_resource
def load_model_single(run_dir: str, device_str: str):
    ckpt_path = os.path.join(run_dir, "best_hybrid7.pt")
    scaler_path = os.path.join(run_dir, "feature_scaler.npz")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Expected best_hybrid7.pt in: {run_dir}")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Expected feature_scaler.npz in: {run_dir}")

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
    return model, class_names, scaler, ckpt_path, device


@st.cache_resource
def load_model_multi(run_dir: str, device_str: str):
    ckpt_path = os.path.join(run_dir, "best_hybrid7_multilabel.pt")
    scaler_path = os.path.join(run_dir, "feature_scaler.npz")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Expected best_hybrid7_multilabel.pt in: {run_dir}")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Expected feature_scaler.npz in: {run_dir}")

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
    return model, class_names, scaler, ckpt_path, device


# ============================================================
# Inference helpers
# ============================================================
def _prep_inputs(
    x: np.ndarray,
    y: np.ndarray,
    scaler,
    n_points: int,
    device: torch.device,
):
    x_res, y_res = resample_curve(x, y, n_points=n_points)
    x_res, y_res = normalize_xy(x_res, y_res)

    X = np.stack([x_res, y_res], axis=0).astype(np.float32)
    xb = torch.from_numpy(X).unsqueeze(0).to(device=device, dtype=torch.float32)

    feat = compute_features_7(x_res, y_res).astype(np.float32).reshape(1, -1)
    mu = np.asarray(scaler["mean"]).reshape(1, -1)
    sd = np.asarray(scaler["std"]).reshape(1, -1)
    sd = np.where(sd < 1e-8, 1.0, sd)
    feat_s = (feat - mu) / sd
    fb = torch.from_numpy(feat_s).to(device=device, dtype=torch.float32)

    return xb, fb


def predict_single(
    model,
    scaler,
    class_names: List[str],
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_points: int = 512,
) -> Dict[str, Any]:
    xb, fb = _prep_inputs(x, y, scaler, n_points, device)
    with torch.no_grad():
        logits = model(xb, fb)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)

    order = np.argsort(-probs)
    i1, i2 = int(order[0]), int(order[1])
    return {
        "mode": "single",
        "scores": probs,
        "pred1_label": class_names[i1],
        "pred1_score": float(probs[i1]),
        "pred2_label": class_names[i2],
        "pred2_score": float(probs[i2]),
        "margin": float(probs[i1] - probs[i2]),
    }


def predict_multi(
    model,
    scaler,
    class_names: List[str],
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_points: int = 512,
) -> Dict[str, Any]:
    xb, fb = _prep_inputs(x, y, scaler, n_points, device)
    with torch.no_grad():
        logits = model(xb, fb)
        scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

    order = np.argsort(-scores)
    i1, i2 = int(order[0]), int(order[1])
    return {
        "mode": "multi",
        "scores": scores,
        "pred1_label": class_names[i1],
        "pred1_score": float(scores[i1]),
        "pred2_label": class_names[i2],
        "pred2_score": float(scores[i2]),
        "margin": float(scores[i1] - scores[i2]),
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


def bar_top2(labels: List[str], vals: List[float], title: str):
    fig = plt.figure(figsize=(6, 2.2))
    plt.barh(labels[::-1], vals[::-1])
    plt.xlabel("Score")
    plt.title(title)
    plt.xlim(0, 1.0)
    st.pyplot(fig)
    plt.close(fig)


def prob_bar_top2(class_names: List[str], scores: np.ndarray, mode: str):
    order = np.argsort(-scores)[:2]
    labels = [class_names[i] for i in order]
    vals = [float(scores[i]) for i in order]
    title = "Top probabilities (Top-2)" if mode == "single" else "Top scores (Top-2, multi-label)"
    bar_top2(labels, vals, title)


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

    st.header("Step 2 – Model artifacts")
    run_dir = st.text_input(
        "Local cache folder (results/...)",
        value=os.path.join("results", "hybrid7_final_v1"),
        help="This folder is used only as a local cache for downloaded artifacts.",
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

    st.header("Step 3 – Decision regime")
    p_strong = st.slider(
        "Single-model confidence threshold (p1)",
        0.50,
        0.99,
        0.90,
        0.01,
        help="If single-label Top-1 probability is below this threshold, the app switches to multi-label model (mixed-regime mode).",
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

# Layout
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Select dynamogram")
    selected_id = st.selectbox("graph_id", graph_ids, index=0)
    st.caption("Single-graph inference (interactive demo).")

# Get curve
try:
    x, y = get_curve(df, selected_id)
except Exception as e:
    st.error(f"Failed to read curve for graph_id={selected_id}: {e}")
    st.stop()

# Download + load both models
try:
    with st.spinner("Downloading model artifacts (private repo)..."):
        downloaded = ensure_private_artifacts(run_dir)

    if downloaded:
        st.caption("Model artifacts downloaded and cached locally.")
    else:
        st.caption("Using cached model artifacts.")

    single_model, class_names_s, scaler, ckpt_single, device = load_model_single(run_dir, device_str)
    multi_model, class_names_m, scaler2, ckpt_multi, device2 = load_model_multi(run_dir, device_str)

    if class_names_s != class_names_m:
        raise RuntimeError("Class list mismatch between single and multi checkpoints.")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Inference: decision regime
try:
    pred_s = predict_single(
        model=single_model,
        scaler=scaler,
        class_names=class_names_s,
        x=x,
        y=y,
        device=device,
        n_points=int(n_points),
    )

    if pred_s["pred1_score"] >= float(p_strong):
        final = pred_s
        used = "single"
        used_ckpt = ckpt_single
    else:
        pred_m = predict_multi(
            model=multi_model,
            scaler=scaler,  # scaler must match features_7 standardization
            class_names=class_names_s,
            x=x,
            y=y,
            device=device2,
            n_points=int(n_points),
        )
        final = pred_m
        used = "multi"
        used_ckpt = ckpt_multi

except Exception as e:
    st.error(f"Inference failed for graph_id={selected_id}: {e}")
    st.stop()

# Extract
l1, p1 = final["pred1_label"], final["pred1_score"]
l2, p2 = final["pred2_label"], final["pred2_score"]
margin = final.get("margin", None)

# Hash for traceability
model_hash = short_file_hash(used_ckpt)

# Left column: output + top2
with col_left:
    st.subheader("Model output")
    st.success(f"**Result:** {l1}  \nScore: **{p1:.2f}**")

    st.caption(
        f"Decision regime: **{used}** · "
        f"threshold(p1): **{p_strong:.2f}** · "
        f"version: **{model_hash}**"
    )
    st.caption(f"Checkpoint: `{os.path.basename(used_ckpt)}`")

    st.markdown("### Top-2")
    prob_bar_top2(class_names_s, final["scores"], mode=final["mode"])

# Right column: plot + export
with col_right:
    st.subheader("Dynamogram plot")
    plot_dynamogram(x, y, title=f"graph_id: {selected_id}")

    st.divider()
    st.subheader("Export prediction (selected graph)")

    out_row = {
        "graph_id": str(selected_id),
        "decision_regime": used,  # single or multi
        "threshold_p1": float(p_strong),

        "pred1_label": l1,
        "pred1_score": float(p1),
        "pred2_label": l2,
        "pred2_score": float(p2),
        "margin": (float(margin) if margin is not None else None),

        "model_single_ckpt": os.path.basename(ckpt_single),
        "model_multi_ckpt": os.path.basename(ckpt_multi),
        "selected_checkpoint": os.path.basename(used_ckpt),
        "selected_checkpoint_hash": model_hash,

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
