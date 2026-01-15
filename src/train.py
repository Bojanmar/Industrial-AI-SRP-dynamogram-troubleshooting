# src/train.py
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from utils import set_seed, ensure_dir, resolve_device
from preprocessing import load_and_prepare_signal
from features import (
    compute_features_7, compute_features_17,
    standardize_features, FEATURE_NAMES_7, FEATURE_NAMES_17
)
from models.cnn1d import Dyn1DCNN
from models.hybrid import HybridModel


class DynDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class HybridDynDataset(Dataset):
    def __init__(self, X: np.ndarray, F: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.F = torch.from_numpy(F)
        self.y = torch.from_numpy(y)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.F[idx], self.y[idx]


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate_cnn(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_pred.append(pred)
        all_true.append(yb.detach().cpu().numpy())
    return total_loss / max(n, 1), np.concatenate(all_true), np.concatenate(all_pred)


@torch.no_grad()
def evaluate_hybrid(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    for xb, fb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        fb = fb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb, fb)
        loss = nn.functional.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_pred.append(pred)
        all_true.append(yb.detach().cpu().numpy())
    return total_loss / max(n, 1), np.concatenate(all_true), np.concatenate(all_pred)


def save_confusion_matrix_png(cm: np.ndarray, labels: list[str], out_path: str, title: str):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    if len(labels) <= 20:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm[i, j]
                if v > 0:
                    plt.text(j, i, str(v), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_feature_matrix(X_all: np.ndarray, feature_mode: str) -> tuple[np.ndarray, list[str]]:
    """
    X_all: [N,2,n_points] where channel0=x_res, channel1=y_res
    feature_mode: "hybrid7" or "hybrid17"
    """
    N = X_all.shape[0]
    if feature_mode == "hybrid7":
        F = np.zeros((N, 7), dtype=np.float32)
        for i in range(N):
            x_res = X_all[i, 0, :]
            y_res = X_all[i, 1, :]
            F[i, :] = compute_features_7(x_res, y_res)
        return F, FEATURE_NAMES_7

    if feature_mode == "hybrid17":
        F = np.zeros((N, 17), dtype=np.float32)
        for i in range(N):
            x_res = X_all[i, 0, :]
            y_res = X_all[i, 1, :]
            F[i, :] = compute_features_17(x_res, y_res)
        return F, FEATURE_NAMES_17

    raise ValueError("feature_mode must be 'hybrid7' or 'hybrid17'")


def save_split_ids(out_dir: str, graph_ids: list[str], idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray):
    def _save(name: str, idxs: np.ndarray):
        pd.DataFrame({"graph_id": [graph_ids[i] for i in idxs]}).to_csv(
            os.path.join(out_dir, f"split_{name}_ids.csv"), index=False
        )
    _save("train", idx_train)
    _save("val", idx_val)
    _save("test", idx_test)


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True, help="Path to CSV with columns: graph_id,x,y,label,(optional point_no)")
    ap.add_argument("--model", choices=["cnn", "hybrid7", "hybrid17"], default="cnn")
    ap.add_argument("--n_points", type=int, default=512)
    ap.add_argument("--min_class_count", type=int, default=10)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cuda/cpu (optional)")
    ap.add_argument("--out_dir", default="results/run", help="Output directory")

    # --- performance flags ---
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (Windows: 0 or 2 recommended)")
    ap.add_argument("--pin_memory", action="store_true", help="Use pin_memory for faster GPU transfers")

    # --- robustness: early stopping + scheduler ---
    ap.add_argument("--early_stopping_patience", type=int, default=7, help="Stop if no val improvement for N epochs")
    ap.add_argument("--early_stopping_min_delta", type=float, default=1e-4, help="Min val macroF1 improvement")
    ap.add_argument("--lr_patience", type=int, default=3, help="ReduceLROnPlateau patience")
    ap.add_argument("--lr_factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    ap.add_argument("--min_lr", type=float, default=1e-6, help="ReduceLROnPlateau min_lr")

    args = ap.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # Save config for reproducibility
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # 1) Load & build signal tensor
    graph_ids, X_all, y_all, le, labels_str = load_and_prepare_signal(
        csv_path=args.csv_path,
        n_points=args.n_points,
        min_class_count=args.min_class_count,
        save_dir=args.out_dir,
    )
    n_classes = len(le.classes_)
    class_names = list(le.classes_)

    # 2) Graph-level split (with saved indices)
    idx_all = np.arange(len(graph_ids))

    # First: test split
    X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        X_all, y_all, idx_all,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_all
    )

    # Second: val split (on trainval)
    val_relative = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_trainval, y_trainval, idx_trainval,
        test_size=val_relative,
        random_state=args.seed,
        stratify=y_trainval
    )

    # Save split IDs (graph_id lists)
    save_split_ids(args.out_dir, graph_ids, idx_train, idx_val, idx_test)

    # Metrics log
    metrics_rows = []

    # 3) Train
    if args.model == "cnn":
        train_loader = make_loader(DynDataset(X_train, y_train), args.batch_size, True, args.num_workers, args.pin_memory)
        val_loader = make_loader(DynDataset(X_val, y_val), args.batch_size, False, args.num_workers, args.pin_memory)
        test_loader = make_loader(DynDataset(X_test, y_test), args.batch_size, False, args.num_workers, args.pin_memory)

        model = Dyn1DCNN(n_classes=n_classes, dropout=args.dropout).to(device)
        class_weights = compute_class_weights(y_train, n_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            
        )

        best_path = os.path.join(args.out_dir, "best_cnn.pt")
        best_val = -1.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, n = 0.0, 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)

            train_loss = total_loss / max(n, 1)

            val_loss, y_true_val, y_pred_val = evaluate_cnn(model, val_loader, device)
            val_f1 = f1_score(y_true_val, y_pred_val, average="macro")

            lr_now = float(optimizer.param_groups[0]["lr"])
            print(f"[CNN] Epoch {epoch:02d}/{args.epochs} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_macroF1={val_f1:.4f}")

            metrics_rows.append({
                "epoch": epoch, "lr": lr_now,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macroF1": val_f1
            })

            scheduler.step(val_f1)

            improved = (val_f1 - best_val) > args.early_stopping_min_delta
            if improved:
                best_val = val_f1
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {"model_state": model.state_dict(), "label_classes": class_names, "args": vars(args)},
                    best_path
                )
                print(f"  ✓ Saved best model: {best_path} (val_macroF1={best_val:.4f})")
            else:
                patience_counter += 1
                print(f"  (no improvement) patience {patience_counter}/{args.early_stopping_patience}")

            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch} val_macroF1={best_val:.4f}")
                break

        # Load best + test
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_loss, y_true, y_pred = evaluate_cnn(model, test_loader, device)

        # Save metrics
        pd.DataFrame(metrics_rows).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)

        # Save outputs
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        cm = confusion_matrix(y_true, y_pred)
        np.save(os.path.join(args.out_dir, "confusion_matrix.npy"), cm)
        save_confusion_matrix_png(cm, class_names, os.path.join(args.out_dir, "confusion_matrix.png"), "Confusion Matrix — CNN (signal only)")

        pred_df = pd.DataFrame({
            "graph_id": [graph_ids[i] for i in idx_test],
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": [class_names[i] for i in y_true],
            "pred_label": [class_names[i] for i in y_pred],
        })
        pred_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

        print("\n==== TEST (CNN) ====")
        print(f"test_loss={test_loss:.4f} | test_macroF1={f1_score(y_true, y_pred, average='macro'):.4f}")
        print(report)

    else:
        # Hybrid features (build once on full set)
        F_all, feat_names = build_feature_matrix(X_all, args.model)

        # IMPORTANT: use the SAME split indices we already created (idx_train/idx_val/idx_test)
        X_train = X_all[idx_train]
        X_val = X_all[idx_val]
        X_test = X_all[idx_test]

        F_train = F_all[idx_train]
        F_val = F_all[idx_val]
        F_test = F_all[idx_test]

        y_train = y_all[idx_train]
        y_val = y_all[idx_val]
        y_test = y_all[idx_test]

        # Standardize features using train stats only
        F_train_s, F_val_s, F_test_s, mu, sd = standardize_features(F_train, F_val, F_test)
        np.savez(os.path.join(args.out_dir, "feature_scaler.npz"), mean=mu, std=sd)

        # Save feature names
        pd.DataFrame({"feature": feat_names}).to_csv(os.path.join(args.out_dir, "feature_names.csv"), index=False)

        train_loader = make_loader(HybridDynDataset(X_train, F_train_s, y_train), args.batch_size, True, args.num_workers, args.pin_memory)
        val_loader = make_loader(HybridDynDataset(X_val, F_val_s, y_val), args.batch_size, False, args.num_workers, args.pin_memory)
        test_loader = make_loader(HybridDynDataset(X_test, F_test_s, y_test), args.batch_size, False, args.num_workers, args.pin_memory)

        model = HybridModel(n_classes=n_classes, n_feat=F_train_s.shape[1], dropout=args.dropout).to(device)
        class_weights = compute_class_weights(y_train, n_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            
        )

        best_path = os.path.join(args.out_dir, f"best_{args.model}.pt")
        best_val = -1.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, n = 0.0, 0

            for xb, fb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                fb = fb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb, fb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)

            train_loss = total_loss / max(n, 1)

            val_loss, y_true_val, y_pred_val = evaluate_hybrid(model, val_loader, device)
            val_f1 = f1_score(y_true_val, y_pred_val, average="macro")

            lr_now = float(optimizer.param_groups[0]["lr"])
            print(f"[{args.model}] Epoch {epoch:02d}/{args.epochs} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_macroF1={val_f1:.4f}")

            metrics_rows.append({
                "epoch": epoch, "lr": lr_now,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macroF1": val_f1
            })

            scheduler.step(val_f1)

            improved = (val_f1 - best_val) > args.early_stopping_min_delta
            if improved:
                best_val = val_f1
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "label_classes": class_names,
                        "args": vars(args),
                        "n_feat": int(F_train_s.shape[1]),
                        "feature_names": feat_names,
                    },
                    best_path
                )
                print(f"  ✓ Saved best model: {best_path} (val_macroF1={best_val:.4f})")
            else:
                patience_counter += 1
                print(f"  (no improvement) patience {patience_counter}/{args.early_stopping_patience}")

            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch} val_macroF1={best_val:.4f}")
                break

        # Load best + test
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_loss, y_true, y_pred = evaluate_hybrid(model, test_loader, device)

        # Save metrics
        pd.DataFrame(metrics_rows).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)

        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        cm = confusion_matrix(y_true, y_pred)
        np.save(os.path.join(args.out_dir, "confusion_matrix.npy"), cm)
        title = "Confusion Matrix — Hybrid (CNN + 7 features)" if args.model == "hybrid7" else "Confusion Matrix — Hybrid+ (CNN + 17 features)"
        save_confusion_matrix_png(cm, class_names, os.path.join(args.out_dir, "confusion_matrix.png"), title)

        pred_df = pd.DataFrame({
            "graph_id": [graph_ids[i] for i in idx_test],
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": [class_names[i] for i in y_true],
            "pred_label": [class_names[i] for i in y_pred],
        })
        pred_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

        print(f"\n==== TEST ({args.model}) ====")
        print(f"test_loss={test_loss:.4f} | test_macroF1={f1_score(y_true, y_pred, average='macro'):.4f}")
        print(report)


if __name__ == "__main__":
    main()
