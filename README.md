# SRP Dynamogram Classification
## End-to-End Industrial AI using Hybrid Deep Learning

---

## Overview

This repository presents a **full end-to-end industrial AI solution** for classification and diagnosis of **Sucker Rod Pump (SRP) dynamometer cards**, combining:

- **Deep learning (CNNs)**
- **Physics-informed, domain-engineered features**
- **Production-oriented inference logic**

The project is intentionally designed around **real operational constraints**, not idealized academic benchmarks.

It demonstrates how machine learning can be made **usable, explainable, and trustworthy** in industrial environments, where data is noisy, labels are imperfect, and decisions must still be made.

> **This is not just a model — it is a complete diagnostic pipeline, from raw data to production-ready inference.**

---
## Live Demo

🔗 **Streamlit Demo:**  
https://<your-streamlit-app>.streamlit.app

The demo allows interactive inspection of individual SRP dynamograms,
Top-2 predictions, and confidence-aware diagnostic behavior.

## Quickstart (Local)

``bash
pip install -r requirements.txt
streamlit r

## Industrial Problem Context

SRP dynamograms are a core diagnostic tool in oil production, used to identify:

- gas interference  
- fluid pound  
- mechanical wear  
- valve leakage  
- rod/tubing interaction  
- normal operating regimes  

In real field operations:

- multiple failure mechanisms may coexist in a single pump cycle,
- operating conditions continuously change,
- historical labels are assigned by experts, often with unavoidable subjectivity.

As a result, **perfect class separation does not exist** in operational data.

This repository **explicitly embraces that reality instead of hiding it**.

---

## Key Design Philosophy

> **Effective industrial ML is not about perfect data —  
> it is about building systems that remain useful despite imperfect data.**

All architectural and modeling choices in this project follow this principle.

---

## End-to-End Project Flow

The repository implements the entire AI lifecycle:

### 1. Data Ingestion
- SRP dynamograms stored in long/tabular format  
  (`graph_id`, `x`, `y`, `label`)

### 2. Signal Preprocessing
- graph-level grouping  
- resampling to fixed resolution  
- normalization for shape robustness  

### 3. Feature Engineering (Domain-Driven)
- physically meaningful shape descriptors  
- symmetry, fill factor, statistical moments  
- higher-order shape complexity indicators  

### 4. Model Training
- CNN-only model (signal-based)
- Hybrid CNN + engineered features
- class-weighted loss
- early stopping
- reproducible data splits

### 5. Inference & Decision Logic
- Top-1 and Top-2 predictions
- confidence-aware interpretation
- ambiguity-aware diagnostics

### 6. Visualization & Validation
- Streamlit-based UI for expert review
- curve inspection
- probability outputs

This structure reflects how **industrial AI systems are actually built and deployed**.

---

## Model Architectures

Three architectures are implemented and evaluated:

### 1) CNN (Signal-Only Baseline)

- 1D CNN trained directly on normalized dynamogram curves
- Learns pure shape representations
- Strong on dominant operational modes
- Limited robustness for overlapping failure mechanisms

---

### 2) Hybrid CNN + 7 Engineered Features

- CNN encoder + compact set of physics-inspired features
- Improved robustness and interpretability
- Better stability on imbalanced datasets

---

### 3) Hybrid CNN + 17 Engineered Features (Production Model)

Extended feature set including:

- higher-order statistics
- asymmetry metrics
- shape complexity indicators
- low-frequency spectral components

This model provides the **best balance between accuracy, macro-F1, and practical usability**.

---

## Why Hybrid Modeling Matters

Pure deep learning models tend to:

- overfit dominant regimes,
- struggle with minority or mixed failure modes,
- behave unpredictably on ambiguous samples.

By injecting domain knowledge explicitly:

- the model becomes more stable,
- predictions become easier to interpret,
- failure cases are easier to reason about.

This hybrid approach reflects how **experienced production engineers actually think** when analyzing dynamograms.

---

## Label Ambiguity & Industrial Reality

Certain SRP conditions (e.g. *Fluid Pound*, *Gas in Pump*, *Rod Tagging*) are physically overlapping phenomena.

In real operations:

- they may appear simultaneously,
- transitions are continuous rather than discrete,
- expert labels reflect dominant interpretation, not strict ground truth.

---

## Important Design Choice

Instead of aggressively cleaning or merging labels to artificially boost metrics, this project:

- preserves label imperfections,
- evaluates model behavior under realistic ambiguity,
- prioritizes robustness and transparency over cosmetic scores.

> Lower metrics on some minority classes are therefore a reflection of **physical reality**, not a modeling failure.

---

## Inference Strategy (Production-Critical)

A key outcome of this project is the realization that:

> **Top-2 predictions are often more valuable than Top-1 predictions in SRP diagnostics.**

The selected production model (`hybrid7_final_v1`) consistently shows that:

- ambiguous cases are surfaced as competing failure modes,
- engineers resolve uncertainty faster using probability-ranked outputs,
- overall decision quality improves even if single-class metrics decrease.

This is exactly how **industrial decision-support systems should behave**.

---

## Results Summary (Representative)

| Model | Accuracy | Macro-F1 |
|------|----------|----------|
| CNN | ~0.77 | ~0.65 |
| Hybrid (7 features) | ~0.83 | ~0.66 |
| Hybrid (17 features) | ~0.81 | ~0.74 |

The **Hybrid-17 model** achieves the best overall balance,  
while **Hybrid-7** often performs best in practical inference scenarios due to clearer Top-2 behavior.

---

## Streamlit Diagnostic Interface

The repository includes a Streamlit application that enables:

- selection of individual dynamograms,
- curve visualization,
- Top-1 and Top-2 prediction inspection,
- probability-based interpretation.

This bridges the gap between **ML output and engineering judgment**, which is essential for production adoption.

---

## Reproducibility & Engineering Quality

The pipeline includes:

- graph-level train/validation/test splits (no leakage),
- fixed random seeds,
- saved label mappings,
- stored feature scalers,
- checkpointed best models,
- fully logged metrics.

Results are **auditable and reproducible**, which is mandatory for industrial AI.

---

## What This Project Demonstrates

This repository demonstrates the ability to:

- deeply understand SRP physics and diagnostics,
- design ML systems around real operational constraints,
- integrate domain knowledge into neural architectures,
- build end-to-end AI pipelines (not isolated models),
- communicate limitations honestly and clearly.

> This is the difference between **academic ML** and **industrial AI**.

---

## Future Production-Oriented Extensions

Potential next steps include:

- hierarchical or multi-label classification,
- uncertainty thresholds and abstention logic,
- expert-in-the-loop learning,
- integration with real-time well monitoring systems,
- deployment as a diagnostic microservice.

These are **system-level decisions**, not simple model tweaks.

---

## Author

**Bojan Martinović**  
PhD – Petroleum Engineering  

**Focus:**  
Production Engineering · Artificial Lift · AI/ML · Digital Oilfield

Developed by an oil & gas production technology professional with extensive experience in SRP diagnostics, field operations, and applied machine learning.

---

## Disclaimer

### What Is Intentionally Not Included

This repository does **not** include:

- raw dynamogram datasets,
- labeled production field data,
- model prediction outputs on proprietary samples,
- train/validation/test split identifiers,
- production-trained model weights.

### Rationale

SRP dynamogram data originates from real oilfield operations and is subject to:

- confidentiality agreements,
- intellectual property constraints,
- operational security requirements.

### Reproducibility

All code, preprocessing logic, feature definitions, and training pipelines are fully provided.  
The system can be reproduced on any dataset following the documented schema.

This reflects standard **industrial AI practices**, where:

- methodology is shared,
- data ownership is protected.
