# Hybrid Model (CNN + 7 Engineered Features) — Production-Oriented

## Model Description
This hybrid architecture combines:
- a CNN-based encoder for curve shape learning,
- 7 physically meaningful engineered features derived from SRP pump mechanics.

This model demonstrated the **best real-world behavior** when evaluated through Top-2 inference and expert review.

## Engineered Features (7)
The feature set captures:
- dynamogram area and fill factor,
- symmetry and imbalance indicators,
- basic statistical descriptors of the load–position relationship.

These features encode **physical pump behavior** that is not reliably learned from raw shape alone.

## Why This Model Is Selected
Although Hybrid-17 achieved slightly higher macro metrics, this model:
- produces more stable Top-2 predictions,
- aligns better with expert reasoning,
- handles ambiguous cases (e.g. Gas vs Fluid Pound) more realistically.

## Inference Strategy
- Top-1 prediction is accepted only above a confidence threshold
- Otherwise, Top-2 predictions are returned for expert review

This mirrors **real operational decision workflows**.

## Production Readiness
✔ Robust to noisy labels  
✔ Interpretable feature contribution  
✔ Stable behavior under class imbalance  
✔ Designed for human-in-the-loop deployment  

## Role in the System
This model represents the **recommended industrial deployment configuration**.
