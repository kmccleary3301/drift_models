# Entrypoint Reproducibility Audit (RUN.md + Environment Fingerprints)

This repo uses a standard “repro artifact bundle” written alongside run outputs:

- `RUN.md` (human-readable run summary)
- `env_snapshot.json` (package + system snapshot)
- `codebase_fingerprint.json` (git + file hash fingerprint)
- `env_fingerprint.json` (small stable fingerprint for quick comparisons)

The goal of this audit is to ensure *major entrypoints* (anything that produces a non-trivial artifact tree under `outputs/`) write the bundle.

## Status (2026-02-14)

| Entrypoint | Bundle written? | Notes |
| --- | --- | --- |
| `scripts/train_latent.py` | Yes | Writes full bundle + `RUN.md` to `--output-dir` |
| `scripts/train_pixel.py` | Yes | Writes full bundle + `RUN.md` to `--output-dir` |
| `scripts/train_mae.py` | Yes | Writes full bundle + `RUN.md` to `--output-dir` |
| `scripts/train_toy.py` | Yes | Writes full bundle + `RUN.md` to `--output-dir` |
| `scripts/sample_latent.py` | Yes | Writes full bundle to `--output-root` |
| `scripts/sample_pixel.py` | Yes | Writes full bundle to `--output-root` |
| `scripts/eval_fid_is.py` | Yes | Writes full bundle to output dir |
| `scripts/cache_reference_stats.py` | Yes | Writes full bundle to output dir |
| `scripts/export_sd_vae_latents_tensor_file.py` | Yes | Writes full bundle to output dir |
| `scripts/eval_alpha_sweep.py` | Yes | Writes full bundle + `RUN.md` to `--output-root` |
| `scripts/eval_last_k_checkpoints.py` | Yes | Writes full bundle + `RUN.md` to `--output-root` |
| `scripts/run_end_to_end_latent_eval.py` | Yes | Writes full bundle + `RUN.md` to `--output-root` |
| `scripts/run_end_to_end_pixel_eval.py` | Yes | Writes full bundle + `RUN.md` to `--output-root` |
| `scripts/run_imagenet_latent_pipeline.py` | Yes | Writes bundle + `RUN.md` to `outputs/logs/` |
| `scripts/run_latent_table8_benchmark.py` | Yes | Writes bundle + `RUN.md` to `--output-root` |
| `scripts/run_queue_hotpath_benchmark.py` | Yes | Writes bundle + `RUN.md` to `--output-root` |
| `scripts/run_generator_arch_ablations.py` | Yes | Writes bundle + `RUN.md` to `--output-dir` |

## Adding bundle support to a new script

Use the following pattern (adapt output dir naming to the script):

```python
from drifting_models.utils import codebase_fingerprint, environment_fingerprint, environment_snapshot, write_json
from drifting_models.utils.run_md import write_run_md

repo_root = Path(__file__).resolve().parents[1]
write_json(out_dir / "env_snapshot.json", environment_snapshot(paths=[out_dir]))
write_json(out_dir / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
write_json(out_dir / "env_fingerprint.json", environment_fingerprint())
write_run_md(out_dir / "RUN.md", {"output_root": str(out_dir), "args": vars(args)})
```
