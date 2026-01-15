# scripts/collect_results.py
import os
import glob
import pandas as pd

RESULTS_DIR = "results"
OUT_SUMMARY = os.path.join(RESULTS_DIR, "summary_runs.csv")
OUT_PER_CLASS_ALL = os.path.join(RESULTS_DIR, "per_class_metrics_all_runs.csv")
OUT_PIVOT_F1 = os.path.join(RESULTS_DIR, "compare_models_per_class_f1.csv")
OUT_PIVOT_SUPPORT = os.path.join(RESULTS_DIR, "compare_models_per_class_support.csv")


def parse_classification_report(path: str):
    """
    Parses sklearn.metrics.classification_report(text output) saved in classification_report.txt.
    Returns:
      - overall: dict with accuracy, macro_f1, weighted_f1, macro_precision, macro_recall, weighted_precision, weighted_recall, support_total
      - per_class: DataFrame with columns [label, precision, recall, f1, support]
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.readlines() if ln.strip()]

    per_rows = []
    overall = {}

    # Identify "accuracy" line and "macro avg" / "weighted avg"
    # Expected formats:
    # accuracy                         0.8460       526
    # macro avg     0.7078    0.8059    0.7414       526
    # weighted avg  0.8960    0.8460    0.8633       526

    def is_overall_line(prefix: str, ln: str) -> bool:
        return ln.startswith(prefix)

    for ln in lines:
        if is_overall_line("accuracy", ln):
            parts = ln.split()
            # ["accuracy", "0.8460", "526"] OR with spacing
            # safest: take last two
            acc = float(parts[-2])
            support = int(float(parts[-1]))
            overall["accuracy"] = acc
            overall["support_total"] = support

        elif is_overall_line("macro avg", ln):
            parts = ln.split()
            # ["macro","avg","0.7078","0.8059","0.7414","526"]
            overall["macro_precision"] = float(parts[-4])
            overall["macro_recall"] = float(parts[-3])
            overall["macro_f1"] = float(parts[-2])
            overall["support_total"] = int(float(parts[-1]))

        elif is_overall_line("weighted avg", ln):
            parts = ln.split()
            overall["weighted_precision"] = float(parts[-4])
            overall["weighted_recall"] = float(parts[-3])
            overall["weighted_f1"] = float(parts[-2])
            overall["support_total"] = int(float(parts[-1]))

    # Per-class lines: everything that is not accuracy/macro/weighted and has 4 numeric fields at end
    # Example:
    # 3. Fluid pound     0.1395    0.3529    0.2000        17
    for ln in lines:
        if ln.startswith(("accuracy", "macro avg", "weighted avg")):
            continue

        parts = ln.split()
        if len(parts) < 5:
            continue

        # last 4 are numeric: precision, recall, f1, support
        try:
            precision = float(parts[-4])
            recall = float(parts[-3])
            f1 = float(parts[-2])
            support = int(float(parts[-1]))
        except ValueError:
            continue

        label = " ".join(parts[:-4])
        per_rows.append(
            {"label": label, "precision": precision, "recall": recall, "f1": f1, "support": support}
        )

    per_class = pd.DataFrame(per_rows)
    return overall, per_class


def infer_model_name(run_name: str) -> str:
    """
    Best-effort model inference from folder name.
    """
    lower = run_name.lower()
    if "hybrid17" in lower:
        return "hybrid17"
    if "hybrid7" in lower:
        return "hybrid7"
    if "cnn" in lower:
        return "cnn"
    return "unknown"


def main():
    run_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "*")) if os.path.isdir(d)]
    run_dirs = sorted(run_dirs)

    summary_rows = []
    per_class_rows = []

    for run_dir in run_dirs:
        run = os.path.basename(run_dir)
        report_path = os.path.join(run_dir, "classification_report.txt")
        if not os.path.isfile(report_path):
            continue

        overall, per_class = parse_classification_report(report_path)

        model = infer_model_name(run)
        summary_rows.append(
            {
                "run": run,
                "model": model,
                "accuracy": overall.get("accuracy"),
                "macro_f1": overall.get("macro_f1"),
                "weighted_f1": overall.get("weighted_f1"),
                "macro_precision": overall.get("macro_precision"),
                "macro_recall": overall.get("macro_recall"),
                "weighted_precision": overall.get("weighted_precision"),
                "weighted_recall": overall.get("weighted_recall"),
                "support_total": overall.get("support_total"),
                "report_path": report_path,
            }
        )

        if not per_class.empty:
            per_class["run"] = run
            per_class["model"] = model
            per_class_rows.append(per_class)

            # Also save ranked table for this run
            ranked = per_class.sort_values(["f1", "support"], ascending=[False, False])
            out_ranked = os.path.join(RESULTS_DIR, f"per_class_ranked_by_f1_{run}.csv")
            ranked.to_csv(out_ranked, index=False)

    if not summary_rows:
        raise RuntimeError(f"No runs with classification_report.txt found under: {RESULTS_DIR}")

    df_summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    df_summary.to_csv(OUT_SUMMARY, index=False)
    print("Saved:", OUT_SUMMARY)

    df_per_all = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()
    df_per_all.to_csv(OUT_PER_CLASS_ALL, index=False)
    print("Saved:", OUT_PER_CLASS_ALL)

    # Pivot comparisons: per-class F1 and support
    if not df_per_all.empty:
        pivot_f1 = df_per_all.pivot_table(index="run", columns="label", values="f1", aggfunc="first")
        pivot_support = df_per_all.pivot_table(index="run", columns="label", values="support", aggfunc="first")

        pivot_f1 = pivot_f1.reset_index()
        pivot_support = pivot_support.reset_index()

        pivot_f1.to_csv(OUT_PIVOT_F1, index=False)
        pivot_support.to_csv(OUT_PIVOT_SUPPORT, index=False)

        print("Saved:", OUT_PIVOT_F1)
        print("Saved:", OUT_PIVOT_SUPPORT)

    print("\n=== Summary (top runs by macro_f1) ===")
    print(df_summary[["run", "model", "accuracy", "macro_f1", "weighted_f1", "support_total"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
