# Feature-Space Drifting Loss Parity (Paper ↔ Repo)

This note maps Appendix A.5/A.6 (“Multi-scale features” + “Feature and drift normalization”) onto the repo implementation, and calls out known deltas that matter for strict paper parity.

Paper scan reference: `Drift_Models/Drift_Models.md` sections:
- **A.5** Multi-scale Features for Drifting Loss
- **A.6** Feature and Drift Normalization

## 1) Paper specification (compressed)

### A.5 feature set
For each ResNet stage feature map `(H_i x W_i x C_i)`, the paper constructs multiple `C_i`-dim vectors:
- (a) per-location vectors (`H_i * W_i` of them)
- (b) global mean + global std (2 vectors)
- (c) patch-2 pooled mean + std (`H_i/2 * W_i/2` each)
- (d) patch-4 pooled mean + std (`H_i/4 * W_i/4` each)

Additionally, on the encoder input `(H_0 x W_0 x C_0)`, it adds a per-channel mean of squared values (`x^2` mean), yielding a `C_0`-dim vector.

The paper states: “All these losses, **in addition to** the vanilla drifting loss without `φ`, are summed.”

### A.6 normalization + multiple temperatures
For each feature vector slot `φ_j ∈ R^{C_j}`:
- Feature scale `S_j = E[||φ_j(x) - φ_j(y)||] / sqrt(C_j)` (batch empirical; stopgrad)
- Temperature scaling: `τ̃_j = τ * sqrt(C_j)` with `τ ∈ {0.02, 0.05, 0.2}`
- Drift normalization `λ_j = sqrt(E[||V_j||^2 / C_j])` (batch empirical)
- Loss uses MSE on normalized feature and normalized drift.

Multiple temperatures are aggregated by summing normalized drifts across `τ`, then using the aggregated drift in the stopgrad target.

## 2) Repo implementation map

### 2.1 Feature extraction and vectorization (A.5)
- Extraction: `drifting_models/features/vectorize.py::extract_feature_maps()`
- Vectorization: `drifting_models/features/vectorize.py::vectorize_feature_maps()`
- Controlled by: `drifting_models/features/vectorize.py::FeatureVectorizationConfig`
  - per-location vectors: `include_per_location`
  - global mean/std: `include_global_stats`
  - patch-2 mean/std: `include_patch2_stats`
  - patch-4 mean/std: `include_patch4_stats`
  - input `x^2` per-channel mean: `include_input_x2_mean`

### 2.2 Feature normalization and drift normalization (A.6)
Implemented in `drifting_models/drift_loss.py::feature_space_drifting_loss()`:
- Feature scale `S_j`:
  - `_normalize_features()` computes `mean(cdist)/sqrt(C)` and detaches the scale.
- Temperature scaling by `sqrt(C)`:
  - `FeatureDriftingConfig.scale_temperature_by_sqrt_channels`
  - `effective_temperature = tau * sqrt(C)` when enabled.
- Drift normalization `λ_j`:
  - `_normalize_drifts()` uses `_drift_scale()` which matches `sqrt(E[||V||^2 / C])` and detaches.
- Location normalization sharing:
  - `FeatureDriftingConfig.share_location_normalization` computes per-location scales, then shares one scale by averaging across locations (paper-feasible interpretation without cross-location pairwise `cdist`).

## 3) Known deltas vs paper (actionable)

### Delta A — “vanilla drifting loss without φ” term
- Paper: explicitly adds the raw drifting loss term in addition to feature-based terms.
- Repo: when feature drifting is enabled (`GroupedDriftStepConfig.feature_config != None`), training uses *only* the feature-space loss path (`drifting_models/train/stage2.py::_feature_group_loss()`).

**Status (2026-02-14)**: implemented as an opt-in flag:
- `FeatureDriftingConfig.include_raw_drift_loss` (+ `raw_drift_loss_weight`)
- CLI: `--feature-include-raw-drift-loss` / `--feature-raw-drift-loss-weight`

When enabled, training logs:
- `feature_loss` (feature-space term)
- `raw_loss` (vanilla drifting loss term)
- `loss` (sum)

### Delta B — multiple-temperature aggregation semantics
- Paper: compute normalized drift per `τ`, then **sum drifts** and apply a single MSE target.
- Repo: currently computes an MSE term per `τ` (and averages via `torch.stack(loss_terms).mean()`).

**Status (2026-02-14)**: both semantics are supported:
- `FeatureDriftingConfig.temperature_aggregation="per_temperature_mse"` (legacy behavior; MSE per temperature)
- `FeatureDriftingConfig.temperature_aggregation="sum_drifts_then_mse"` (**paper-style**; sum normalized drifts, then one MSE)

Strict-parity configs/runbooks should use `sum_drifts_then_mse` (CLI: `--feature-temperature-aggregation sum_drifts_then_mse`).

## 4) What is already validated by tests
- Backprop through feature-space drifting: `tests/unit/test_feature_drift_loss.py`
- Numeric stability under autocast regimes: `tests/unit/test_numeric_stability.py`
