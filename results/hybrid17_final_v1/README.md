# Hybrid+ Model (CNN + 17 Engineered Features)

## Model Description
This model extends the Hybrid-7 architecture with an enriched feature space,
including higher-order statistics and shape complexity descriptors.

## Feature Set (17)
In addition to the base physical features, the model includes:
- skewness and kurtosis,
- slope asymmetry indicators,
- low-frequency FFT components,
- curve complexity measures.

## Performance Characteristics
- Highest macro-F1 under controlled evaluation
- Strong discrimination for dominant classes
- Increased sensitivity to feature scaling and label noise

## Trade-offs
While statistically strong, the model:
- is more sensitive to imperfect labels,
- shows less stable Top-2 behavior in ambiguous regimes,
- requires stricter data consistency.

## Intended Use
This model is best suited for:
- offline analysis,
- research benchmarking,
- feature relevance studies.

It is **not the primary production recommendation**.
