# CNN Model — Signal-Only Baseline

## Model Description
This model is a 1D Convolutional Neural Network trained directly on normalized SRP dynamometer curves.
It serves as a **signal-only baseline**, relying purely on shape information without any engineered physical features.

## Purpose
The CNN baseline is used to:
- establish a lower-bound performance reference,
- evaluate how much information can be extracted from raw curve geometry alone,
- identify failure modes that require domain knowledge beyond shape similarity.

## Input
- Resampled dynamogram curve
- Shape: [2 × 512] (normalized x and y)

## Output
- Single-label class prediction
- Softmax probabilities per class

## Observed Behavior
- Strong performance on dominant operational regimes (e.g. *Flowing through pump*)
- Limited discrimination between physically overlapping failure modes
- High confusion for classes with similar geometric signatures

## Limitations
- No physical context
- Sensitive to label ambiguity
- Not suitable as a standalone production diagnostic model

## Role in the Pipeline
This model is intentionally retained to demonstrate:
- the limits of pure deep learning,
- the necessity of domain-informed hybrid approaches in industrial AI systems.
