# Experimental scripts

This folder contains non-default research tooling that is intentionally separated from stable claim-facing entrypoints.

## Subfolders

- `ablations/`: architecture and loss ablation runners.
- `benchmarks/`: performance micro-benchmarks and throughput probes.
- `checks/`: focused validation checks for specific toggles.
- `eval/`: non-default evaluation sweeps and checkpoint trend tools.
- `pipelines/`: higher-level orchestration scripts and package builders.
- `recovery/`: run recovery summaries and restart helpers.

## Compatibility behavior

Most moved scripts keep their original top-level wrapper under `scripts/` so older commands still work.

Examples:

- `scripts/run_generator_arch_ablations.py` forwards to `scripts/experimental/ablations/generator_arch.py`
- `scripts/run_feature_drift_kernel_compile_benchmark.py` forwards to `scripts/experimental/benchmarks/feature_drift_kernel_compile.py`
- `scripts/summarize_recovery_matrix_2x2.py` forwards to `scripts/experimental/recovery/summarize_matrix_2x2.py`
- `scripts/eval_alpha_sweep.py` forwards to `scripts/experimental/eval/alpha_sweep.py`
- `scripts/run_end_to_end_latent_eval.py` forwards to `scripts/experimental/pipelines/end_to_end_latent_eval.py`

For new docs and automation, prefer the canonical paths in `scripts/experimental/`.

Deprecated compatibility wrappers are listed in `docs/deprecation_matrix.md` and emit migration warnings at runtime.
