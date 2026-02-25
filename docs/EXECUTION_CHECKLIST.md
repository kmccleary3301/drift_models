# Execution Checklist (A–K)

This checklist tracks the incremental A–K execution plan used to operationalize the broader `IMPLEMENTATION_PLAN.md`.

## A. Sampling + artifacts
- [x] Add reusable sampling utilities (`drifting_models/sampling/`)
- [x] Implement pixel sampler script (`scripts/sample_pixel.py`)
- [x] Implement latent sampler script (`scripts/sample_latent.py`)
- [x] Add sampler tests (unit + integration)

## B. Checkpoint parity for sampling
- [x] Save `extra.model_config` in training checkpoints
- [x] Save `extra.config_hash` for config mismatch guards
- [x] Add checkpoint loader helper + tests

## C. Evaluation robustness + caching
- [x] Improve `scripts/eval_fid_is.py` error handling and protocol fields
- [x] Add reference stats cache (`--cache-reference-stats` / `--load-reference-stats`)
- [x] Add ImageFolder extension filtering
- [x] Add integration tests for caching and filtering

## D. End-to-end pixel runner
- [x] Implement `scripts/run_end_to_end_pixel_eval.py` (train -> sample -> eval)
- [x] Add integration test for runner

## E. CIFAR-10 export + configs + dataset fingerprints
- [x] Implement `scripts/export_cifar10_imagefolder.py`
- [x] Add CIFAR configs under `configs/pixel/`
- [x] Record real-batch provider fingerprints in summaries when queue is enabled

## F. Pixel MAE export pipeline
- [x] Fix config coercion for `None` defaults in train scripts
- [x] Implement MAE export pipeline runner + smoke test

## G. End-to-end latent runner
- [x] Implement `scripts/run_end_to_end_latent_eval.py` (train -> sample -> eval)
- [x] Add integration test for latent runner

## H. Alpha sweep evaluation
- [x] Define alpha sweep protocol (per-alpha sampling + eval with shared reference cache)
- [x] Implement `scripts/eval_alpha_sweep.py` (JSON + MD outputs)
- [x] Add integration smoke test

## I. Last-K checkpoint evaluation
- [x] Add step checkpoint directory saving (`--checkpoint-dir`) to train scripts
- [x] Implement checkpoint selection utilities (`drifting_models/utils/checkpoints.py`)
- [x] Implement `scripts/eval_last_k_checkpoints.py` (JSON + MD outputs)
- [x] Add integration smoke test

## J. Runbook ops
- [x] Write per-run `RUN.md` for runners/evaluators (`drifting_models/utils/run_md.py`)
- [x] Implement experiment log appender (`scripts/append_experiment_log.py`)
- [x] Add optional `--append-experiment-log` hooks in runners/evaluators

## K. Reproduction report
- [x] Draft `docs/reproduction_report.md`
- [x] Add claim-to-evidence map (tests as evidence)
- [x] Add explicit gaps and next experiments

