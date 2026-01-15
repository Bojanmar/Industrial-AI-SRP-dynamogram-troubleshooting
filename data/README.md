# Dataset Specification
## SRP Dynamogram Tabular Format (Production-Oriented)

---

## Purpose

This repository uses a **long / tabular representation** of **Sucker Rod Pump (SRP) dynamometer cards**, where each dynamogram is stored as a set of sampled curve points.

This format is intentionally chosen to support:

- **reproducible preprocessing**
- **physics-informed feature engineering**
- **hybrid deep learning models**
- **production-grade inference pipelines**

The dataset structure reflects **real industrial data flows**, not academic toy datasets.

---

## Data Representation Concept

Each dynamogram represents **one pump cycle** and is identified by a unique `graph_id`.

Each row in the dataset corresponds to **one sampled point** of that dynamogram curve.

One dynamogram → many rows
One graph_id → one physical pump condition

This representation enables:

- graph-level splitting (**no data leakage**)
- flexible resampling resolutions
- robust normalization independent of original sampling density

---

## Required Columns

The dataset **must** contain the following columns:

---

### 1) `graph_id` *(string)*

Unique identifier of a single SRP dynamogram.

All rows belonging to the same dynamogram **must share the same `graph_id`**.

#### Examples
Gas in pump/10032

Fluid pound/151
Flowing through pump/87231

**Best practices:**

- `graph_id` should be **stable and unique across time**
- avoid reusing IDs after relabeling or reprocessing

---

### 2) `x` *(float)*

Horizontal axis of the dynamogram curve.

- Represents normalized stroke position or displacement
- Recommended range: **[0.0 – 1.0]**

If not normalized, preprocessing will normalize after resampling.

---

### 3) `y` *(float)*

Vertical axis of the dynamogram curve.

- Represents load / force / torque proxy
- Preprocessing normalizes amplitude to preserve **shape invariance**
- Absolute magnitude is intentionally **not required**

---

### 4) `label` *(string)*

Class label assigned to the **entire dynamogram**.

This is a **graph-level label**, repeated across all points belonging to the same `graph_id`.

#### Examples
Gas in pump
Fluid pound
Flowing throught pump
Pump wear
TV leaking


⚠️ **Important assumption:**  
The pipeline assumes **single-label classification per graph**, even though multiple physical phenomena may coexist.

This reflects how SRP data is labeled in **real operations**.

---

## Optional Columns

### `point_no` *(integer)*

Optional point ordering index.

- If present → points are sorted by `point_no`
- If absent → points are sorted by `x`

**Recommended when:**

- original sampling is irregular
- data originates from image digitization
- multiple resampling passes are expected

---

## Graph-Level Constraints

To ensure stable training and inference:

- Each `graph_id` should contain **at least 50 points**  
  (more points improve resampling stability)

- No missing values are allowed in:
  - `graph_id`
  - `x`
  - `y`
  - `label`

- Each `graph_id` must map to **exactly one label**

---

## Label Quality & Industrial Reality

SRP dynamogram labels in operational datasets are **not ground truth** in a strict physical sense.

Common sources of ambiguity include:

- simultaneous failure mechanisms (e.g. *Gas + Fluid Pound*)
- transitional operating regimes
- subjective expert interpretation
- historical label drift

This dataset specification **does not attempt to eliminate these ambiguities**.

Instead, the modeling pipeline is designed to:

- remain robust under label noise
- surface uncertainty through **Top-2 predictions**
- support **expert-in-the-loop** review

> This is a deliberate **industrial design choice**, not a limitation.

---

## Class Imbalance Considerations

Operational SRP datasets are typically **highly imbalanced**:

- normal / flowing conditions dominate
- severe failures are rare
- tagging-related events are often underrepresented

The training pipeline therefore includes:

- graph-level stratified splitting
- class-weighted loss functions
- macro-F1 evaluation metrics

⚠️ Users are strongly discouraged from oversampling or synthetic balancing **without domain validation**.

---

## Train / Validation / Test Splitting

All dataset splits are performed **at the graph level**:

- no points from the same `graph_id` appear in multiple sets
- leakage between train and test is fully avoided
- generalization reflects **real deployment conditions**

> This is **non-negotiable** for industrial ML.

---

## Recommended Production Usage

For real-world deployment and expert workflows:

- use **Top-2 predictions with probabilities**, not Top-1 only
- define confidence thresholds for automatic vs. manual review
- route low-margin cases to expert validation
- monitor class-level confusion over time

This dataset format is designed to support **all of the above**.

---

## Example (Simplified)

```csv
graph_id,x,y,label
2. Gas in pump/10032,0.00,0.12,Gas in pump
2. Gas in pump/10032,0.01,0.18,Gas in pump
2. Gas in pump/10032,0.02,0.25,Gas in pump
...

## Key Takeaway

This dataset specification is built for industrial diagnostics, not academic benchmarks.

It prioritizes:

    robustness over perfection

    interpretability over cosmetic metrics

    practical decision support over theoretical purity

Disclaimer

The original field dataset is not included due to confidentiality constraints.

Any dataset following this specification can be used to fully reproduce the pipeline.