# Performance / Scaling Notes (Grouped Drifting)

This note captures the dominant compute/memory scaling drivers for the grouped drifting training step.
It is intended as an engineering guide for “safe knob ranges” and for diagnosing stalls/OOMs.

## 1) Where the time goes (high level)

In the core grouped step (`drifting_models/train/stage2.py::grouped_drift_training_step`), the drift computation is dominated by:
- pairwise distances (`torch.cdist`) between generated negatives and positive samples, and
- softmax normalizations over the resulting logit matrices.

When feature-space drifting is enabled, the same pattern repeats over:
- feature stages/keys,
- vector slots per stage (depending on vectorization settings), and
- drift temperatures (if multiple are used).

## 2) Main scaling knobs

Let:
- `G` = groups (`--groups`)
- `Nneg` = negatives per group (`--negatives-per-group`)
- `Npos` = positives per group (`--positives-per-group`)
- `U` = unconditional negatives per group (`--unconditional-per-group`, only if queue/CFG is enabled)

### 2.1 Drift-field affinity compute (pixel/latent drifting without feature encoder)

Per group, affinity matrices scale roughly as:
- distance compute: `cdist([Nneg, D], [Npos, D])` and `cdist([Nneg, D], [Nneg+U, D])`
- logit matrices: `[Nneg, Npos]` and `[Nneg, Nneg+U]`

Total work scales approximately linearly in `G` and quadratically in the within-group sample counts:
- `O(G * Nneg * (Npos + Nneg + U))`

Memory footprint for the logits/affinities is similarly proportional to:
- `G * Nneg * (Npos + Nneg + U)` (times element size)

### 2.2 `normalize_over_x` overhead

If `DriftFieldConfig.normalize_over_x=True`, we compute both:
- row-wise softmax (`dim=-1`), and
- column-wise softmax (`dim=-2`),
then combine with a geometric mean.

This improves stability in many regimes but increases softmax work and memory traffic.

### 2.3 Feature-space drifting overhead

For feature drifting (`--use-feature-loss`), cost multiplies by:
- number of feature keys/stages emitted by the encoder, and
- number of vector slots per key after vectorization,
- number of drift temperatures (`--feature-temperatures`).

As a result, a “paper-faithful” multi-stage feature configuration can be *significantly* more expensive than raw pixel/latent drifting.

## 3) Practical guidelines

- If you see **OOM** or extreme slowdowns, reduce in this order:
  1) `--groups`
  2) `--negatives-per-group`
  3) `--positives-per-group`
  4) feature-vectorization density (disable patch/global stats; reduce stages)
  5) number of temperatures (train/eval)
- Keep `--real-num-workers 0` for tensor-shard ingestion (avoids shard-cache thrash).
- For imagefolder/webdataset ingestion, tune loader knobs explicitly:
  - `--disable-real-shuffle` (deterministic order),
  - `--real-pin-memory`,
  - `--real-persistent-workers` + `--real-prefetch-factor` (requires `--real-num-workers > 0`).
- Prefer immutable step checkpoints (`checkpoint_step_*.pt`) for evaluation triggers; mutable `checkpoint.pt` can race.

## 4) What to log/monitor (minimum)

From `latent_summary.json` / `pixel_summary.json`, the following are the fastest indicators:
- `perf.mean_step_time_s`
- `perf.max_peak_cuda_mem_mb`
- `generated_images_per_sec`
- queue underflow/backfill counters (if queue enabled)

## 5) Reproducible perf command set

- Queue hot-path microbenchmark (fast sanity + queue underflow counters):
  - `uv run python scripts/run_queue_hotpath_benchmark.py --device cpu --steps 10 --output-root outputs/benchmarks/queue_hotpath_p2_refine`
- Feature-drift kernel benchmark (legacy slot-loop vs vectorized slot-batch):
  - `uv run python scripts/run_feature_drift_vectorization_benchmark.py --device auto --iterations 40 --warmup 10 --output outputs/benchmarks/feature_drift_vectorization/p2_vectorized_baseline.json`
  - `uv run python scripts/run_feature_drift_vectorization_benchmark.py --device auto --iterations 40 --warmup 10 --disable-normalize-over-x --output outputs/benchmarks/feature_drift_vectorization/p2_vectorized_no_xnorm.json`
- Feature-drift temperature reuse benchmark (per-temp loop vs shared-distance path):
  - `uv run python scripts/run_feature_drift_temperature_reuse_benchmark.py --device auto --iterations 40 --warmup 10 --output outputs/benchmarks/feature_drift_temperature_reuse/p2_temperature_reuse_baseline.json`
  - `uv run python scripts/run_feature_drift_temperature_reuse_benchmark.py --device auto --iterations 40 --warmup 10 --disable-normalize-over-x --output outputs/benchmarks/feature_drift_temperature_reuse/p2_temperature_reuse_no_xnorm.json`
- Feature-drift kernel compile benchmark (eager vs compiled multi-temp kernel):
  - `uv run python scripts/run_feature_drift_kernel_compile_benchmark.py --device auto --iterations 40 --warmup 10 --output outputs/benchmarks/feature_drift_kernel_compile/p2_kernel_compile.json`
- Table-8 proxy benchmark (step time + throughput + memory under target configs):
  - `uv run python scripts/run_latent_table8_benchmark.py --steps 20 --output-root outputs/imagenet/benchmarks/table8_latent_proxy`
- For GPU timing consistency, pin device explicitly with `--device cuda:1` on training/benchmark entrypoints.
- Compile scope policy: only generator `forward` is compiled (`model.forward = torch.compile(...)`); queue/provider, logging, checkpointing, and eval remain eager.

## 6) Latest vectorization benchmark snapshot (2026-02-19)

- Source files:
  - `outputs/benchmarks/feature_drift_vectorization/p2_vectorized_baseline.json`
  - `outputs/benchmarks/feature_drift_vectorization/p2_vectorized_no_xnorm.json`
- Shape: generated `[64,16,192]`, positives `[128,16,192]`, negatives `[96,16,192]`.
- `normalize_over_x=true`: legacy `12.54ms` vs vectorized `0.37ms` (`~33.46x`), max abs diff `3.50e-05`.
- `normalize_over_x=false`: legacy `9.70ms` vs vectorized `0.29ms` (`~33.29x`), max abs diff `4.50e-05`.
- Temperature reuse snapshot:
  - `normalize_over_x=true`: legacy `1.16ms` vs reuse `0.68ms` (`~1.71x`), max abs diff `0`.
  - `normalize_over_x=false`: legacy `1.11ms` vs reuse `0.60ms` (`~1.86x`), max abs diff `0`.

## 7) Stage-1 compile parity snapshot (2026-02-19)

- Runs:
  - eager: `outputs/benchmarks/compile_stage1/eager/latent_summary.json`
  - compiled: `outputs/benchmarks/compile_stage1/compiled/latent_summary.json`
  - parity compare: `outputs/benchmarks/compile_stage1/parity_compare.json`
- Parity result (selected stats): max abs diff `1.82e-12` (numerically matched).
- Observed tradeoff: first compiled step has large warmup overhead on CPU (`~35s`), then steady-state step time returns to sub-20ms range.

## 8) Stage-2 drift-kernel compile snapshot (2026-02-19)

- Artifact: `outputs/benchmarks/feature_drift_kernel_compile/p2_kernel_compile.json`
- Configuration: temperatures `[0.02, 0.05, 0.2]`, normalize-over-x enabled, sqrt-channel temp scaling enabled.
- Results:
  - eager steady-state: `~0.63ms`
  - compiled first-call warmup: `~4878ms`
  - compiled steady-state: `~0.19ms`
  - steady-state speedup: `~3.29x`
  - max abs diff vs eager: `~7.27e-06`
- Caveat: compile path can emit Inductor softmax/matmul precision warnings depending on host defaults.
