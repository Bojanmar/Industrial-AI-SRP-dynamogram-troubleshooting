# src/compare_models.py
import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score


def load_singlelabel_from_predictions(run_dir: str):
    path = os.path.join(run_dir, "test_predictions.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if "y_true" in df.columns and "y_pred" in df.columns:
        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        return {
            "test_macroF1": float(f1_score(y_true, y_pred, average="macro")),
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
        }
    return None


def load_summary(run_dir: str):
    s = os.path.join(run_dir, "summary.json")
    if os.path.exists(s):
        with open(s, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="List of run dirs, e.g. results/run_cnn results/run_tabular results/run_multilabel")
    ap.add_argument("--out_csv", default="results/compare_summary.csv")
    args = ap.parse_args()

    rows = []

    for run_dir in args.runs:
        run_dir = run_dir.rstrip("/\\")
        row = {"run_dir": run_dir}

        summary = load_summary(run_dir)
        if summary is not None:
            row["run_type"] = summary.get("run_type", "")
            row["model"] = summary.get("model", summary.get("best_model", ""))
            row["n_classes"] = summary.get("n_classes", None)
            row["n_graphs_total"] = summary.get("n_graphs_total", None)

            metrics = summary.get("metrics", {})
            # flatten metrics
            for k, v in metrics.items():
                row[k] = v

            # also store best-val if present
            if "best_val_macroF1" in summary:
                row["best_val_macroF1"] = summary["best_val_macroF1"]
            if "best_model" in summary:
                row["best_model"] = summary["best_model"]

        else:
            # fallback: infer from test_predictions
            sl = load_singlelabel_from_predictions(run_dir)
            row["run_type"] = "singlelabel_fallback"
            if sl:
                row.update(sl)
            else:
                row["error"] = "No summary.json and could not parse test_predictions.csv"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort: prefer macroF1-like columns if present
    sort_cols = [c for c in ["test_macroF1", "test_thr_macroF1", "test_topk_macroF1"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols[0], ascending=False)

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("Saved:", args.out_csv)
    print(df)


if __name__ == "__main__":
    main()
