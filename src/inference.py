# src/inference.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import torch


@dataclass
class PredRow:
    graph_id: str
    pred1_label: str
    pred1_prob: float
    pred2_label: Optional[str] = None
    pred2_prob: Optional[float] = None
    margin: Optional[float] = None
    output_mode: str = "top1_only"  # "top1_only" or "top2_shown"


@torch.no_grad()
def predict_with_threshold(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_names: List[str],
    graph_ids: List[str],
    top2_threshold: float = 0.70,
) -> List[PredRow]:
    """
    Returns per-graph predictions using:
      - if top1_prob >= top2_threshold -> output only top1
      - else -> output top1 + top2 + margin
    Assumes loader yields either:
      (xb, yb) for CNN
      or (xb, fb, yb) for hybrid
    Also assumes loader order matches graph_ids order.
    """
    model.eval()

    rows: List[PredRow] = []
    idx_ptr = 0

    for batch in loader:
        # Support CNN loader: (xb, yb) or Hybrid loader: (xb, fb, yb)
        if len(batch) == 2:
            xb, _ = batch
            xb = xb.to(device)
            logits = model(xb)
        else:
            xb, fb, _ = batch
            xb = xb.to(device)
            fb = fb.to(device)
            logits = model(xb, fb)

        probs = torch.softmax(logits, dim=1).cpu().numpy()  # [B, C]
        B = probs.shape[0]

        for i in range(B):
            gid = graph_ids[idx_ptr]
            idx_ptr += 1

            p = probs[i]
            top_idx = np.argsort(-p)[:2]
            i1, i2 = int(top_idx[0]), int(top_idx[1])

            p1 = float(p[i1])
            p2 = float(p[i2])

            if p1 >= top2_threshold:
                rows.append(
                    PredRow(
                        graph_id=gid,
                        pred1_label=class_names[i1],
                        pred1_prob=p1,
                        pred2_label=None,
                        pred2_prob=None,
                        margin=None,
                        output_mode="top1_only",
                    )
                )
            else:
                rows.append(
                    PredRow(
                        graph_id=gid,
                        pred1_label=class_names[i1],
                        pred1_prob=p1,
                        pred2_label=class_names[i2],
                        pred2_prob=p2,
                        margin=p1 - p2,
                        output_mode="top2_shown",
                    )
                )

    return rows
