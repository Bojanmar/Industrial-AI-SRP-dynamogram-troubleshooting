# ============================================================
# SRP Dynamogram Diagnostic Demo (Streamlit)
# Decision regime:
#   - run SINGLE first
#   - if p1 >= threshold -> show SINGLE
#   - else -> run MULTI and show MULTI
# Hybrid CNN + 7 Engineered Features
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
    "Industrial AI demo for SRP dynamogram diagnostics using a Hybrid CNN + 7 domain-engineered features."
)

with st.expander("ℹ️ What this demo shows / does not show", expanded=True):
    st.markdown(
        """
**This demo shows**
- End-to-end inference on SRP-like dynamogram data (tabular long format)
- Resampling + normalization for shape robustness
- Hybrid model: CNN shape encoder + 7 physics-informed engineered features
- **Decision regime:** single-label when confident, multi-label when ambiguous
- **Top-2 ranking** for decision support

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

    if "point_no" in g.columns:
        g = g.sort_values("point_no")
    else:
        g = g.sort_values("x")

    return g["x"].to_numpy(dtype=np.float64), g["y"].to_numpy(dtype=np.float64)


def short_file_hash(path: str, n: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# ============================================================
# GitHub private artifacts download
# ============================================================
def _github_download_file(repo_full: str, path_in_repo: str, token: str) -> bytes:
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


def ensure_private_artifacts_dual(local_run_dir: str) -> bool:
    """
    Downloads BOTH single + multi checkpoints + scaler into local_run_dir (cached).
    Local filenames are fixed:
      - best_hybrid7.pt
      - best_hybrid7_multilabel.pt
      - feature_scaler.npz
    """
    token = st.secrets.get("GITHUB_TOKEN", None)
    repo = st.secrets.get("PRIVATE_REPO", None)

    single_pt_path = st.secrets.get("MODEL_SINGLE_PT_PATH", "best_hybrid7.pt")
    multi_pt_path = st.secrets.get("MODEL_MULTI_PT_PATH", "best_hybrid7_multilabel.pt")
    npz_path = st.secrets.get("SCALER_NPZ_PATH", "feature_scaler.npz")

    if not token or not repo:
        raise RuntimeError("Missing Streamlit Secrets: GITHUB_TOKEN and/or PRIVATE_REPO.")

    run_dir = Path(local_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    local_single = run_dir / "best_hybrid7.pt"
    local_multi = run_dir / "best_hybrid7_multilabel.pt"
    local_npz = run_dir / "feature_scaler.npz"

    downloaded = False

    if not local_single.exists():
        local_single.write_bytes(_github_download_file(repo, single_pt_path, token))
        downloaded = True

    if not local_multi.exists():
        local_multi.write_bytes(_github_download_file(repo, multi_pt_path, token))
        downloaded = True

    if not local_npz.exists():
        local_npz.write_bytes(_github_download_file(repo, npz_path, token))
        downloaded = True

    return downloaded


# ============================================================
# Model loading
# ============================================================
@st.cache_resource
def load_model_from_ckpt(ckpt_path: str, scaler_path: str, device_str: str):
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
    return model, class_names, scaler, device


# ============================================================
# Inference helpers
# ============================================================
def _make_inputs(
    scaler,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_points: int,
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


def _top2_from_scores(class_names: List[str], scores: np.ndarray) -> Dict[str, Any]:
    order = np.argsort(-scores)
    i1, i2 = int(order[0]), int(order[1])
    return {
        "scores": scores,
        "pred1_label": class_names[i1],
        "pred1_prob": float(scores[i1]),
        "pred2_label": class_names[i2],
        "pred2_prob": float(scores[i2]),
        "margin": float(scores[i1] - scores[i2]),
    }


def predict_single(
    model,
    scaler,
    class_names: List[str],
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_points: int,
) -> Dict[str, Any]:
    xb, fb = _make_inputs(scaler, x, y, device, n_points)
    with torch.no_grad():
        logits = model(xb, fb)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
    out = _top2_from_scores(class_names, probs)
    out["mode"] = "single"
    return out


def predict_multi(
    model,
    scaler,
    class_names: List[str],
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_points: int,
) -> Dict[str, Any]:
    xb, fb = _make_inputs(scaler, x, y, device, n_points)
    with torch.no_grad():
        logits = model(xb, fb)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    out = _top2_from_scores(class_names, probs)
    out["mode"] = "multi"
    return out


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


def prob_bar_top2(class_names: List[str], scores: np.ndarray, title: str):
    order = np.argsort(-scores)[:2]
    labels = [class_names[i] for i in order]
    vals = [float(scores[i]) for i in order]

    fig = plt.figure(figsize=(6, 2.2))
    plt.barh(labels[::-1], vals[::-1])
    plt.xlabel("Score")
    plt.title(title)
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

    st.header("Step 2 – Model artifacts")
    run_dir = st.text_input(
        "Local cache folder (results/...)",
        value=os.path.join("results", "hybrid7_final_v1"),
        help="Artifacts are downloaded here and cached.",
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
    threshold = st.slider(
        "Single-model confidence threshold (p1)",
        0.50,
        0.99,
        0.90,
        0.01,
        help="If single-label Top-1 confidence is below this threshold, the app switches to multi-label for ambiguity handling.",
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

# Download + load BOTH models
try:
    with st.spinner("Downloading model artifacts..."):
        downloaded = ensure_private_artifacts_dual(run_dir)

    st.caption("Model artifacts downloaded and cached locally." if downloaded else "Using cached model artifacts.")

    single_ckpt = os.path.join(run_dir, "best_hybrid7.pt")
    multi_ckpt = os.path.join(run_dir, "best_hybrid7_multilabel.pt")
    scaler_path = os.path.join(run_dir, "feature_scaler.npz")

    if not os.path.isfile(single_ckpt):
        raise FileNotFoundError(f"Missing single checkpoint: {single_ckpt}")
    if not os.path.isfile(multi_ckpt):
        raise FileNotFoundError(f"Missing multi checkpoint: {multi_ckpt}")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    model_single, classes_single, scaler, device = load_model_from_ckpt(single_ckpt, scaler_path, device_str)
    model_multi, classes_multi, _, _ = load_model_from_ckpt(multi_ckpt, scaler_path, device_str)

    hash_single = short_file_hash(single_ckpt)
    hash_multi = short_file_hash(multi_ckpt)

except Exception as e:
    st.error(f"Failed to load model(s): {e}")
    st.stop()

st.caption(f"Artifacts cache: `{run_dir}`")
st.caption(f"Single checkpoint: `{os.path.basename(single_ckpt)}` · Multi checkpoint: `{os.path.basename(multi_ckpt)}`")

# Layout
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Select dynamogram")
    selected_id = st.selectbox("graph_id", graph_ids, index=0)
    st.caption("Single-graph inference (interactive demo).")

# Inference + decision regime
try:
    x, y = get_curve(df, selected_id)

    pred_s = predict_single(
        model=model_single,
        scaler=scaler,
        class_names=classes_single,
        x=x, y=y,
        device=device,
        n_points=int(n_points),
    )

    if float(pred_s["pred1_prob"]) >= float(threshold):
        pred = pred_s
        decision_regime = "single (confident)"
        used_mode = "single-label"
        used_checkpoint = os.path.basename(single_ckpt)
        used_hash = hash_single
        used_classes = classes_single
        used_scores = pred["scores"]
        bar_title = "Top scores (Top-2, single-label)"
    else:
        pred_m = predict_multi(
            model=model_multi,
            scaler=scaler,
            class_names=classes_multi,
            x=x, y=y,
            device=device,
            n_points=int(n_points),
        )
        pred = pred_m
        decision_regime = "multi (ambiguity handling)"
        used_mode = "multi-label"
        used_checkpoint = os.path.basename(multi_ckpt)
        used_hash = hash_multi
        used_classes = classes_multi
        used_scores = pred["scores"]
        bar_title = "Top scores (Top-2, multi-label)"

except Exception as e:
    st.error(f"Inference failed for graph_id={selected_id}: {e}")
    st.stop()

l1, p1 = pred["pred1_label"], float(pred["pred1_prob"])
l2, p2 = pred["pred2_label"], float(pred["pred2_prob"])
margin = float(pred["margin"])

# LEFT
with col_left:
    st.subheader("Model output")
    st.success(f"**Result:** {l1}  \nScore: **{p1:.2f}**")

    st.caption(
        f"Decision regime: **{decision_regime}** · threshold(p1): **{threshold:.2f}** · "
        f"mode: **{used_mode}** · version: **{used_hash}**"
    )
    st.caption(f"Checkpoint: `{used_checkpoint}`")

    st.markdown("### Top-2")
    prob_bar_top2(used_classes, used_scores, title=bar_title)

# RIGHT
with col_right:
    st.subheader("Dynamogram plot")
    plot_dynamogram(x, y, title=f"graph_id: {selected_id}")

    st.divider()
    st.subheader("Export prediction (selected graph)")

    out_row = {
        "graph_id": str(selected_id),
        "decision_regime": decision_regime,
        "mode": used_mode,
        "pred1_label": l1,
        "pred1_score": p1,
        "pred2_label": l2,
        "pred2_score": p2,
        "margin": margin,
        "threshold": float(threshold),
        "checkpoint": used_checkpoint,
        "model_version": used_hash,
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
