# Faithfulness Evidence Requirements

This document defines the minimum artifact bundle required to mark paper-facing faithfulness gates as complete.

## 1) MAE pretrain evidence (required before latent faithful claims)

For each MAE encoder used in a paper-facing latent run, require all of:

- `mae_summary.json` with:
  - `train_config.encoder_arch` (must be `paper_resnet34_unet` for faithful templates),
  - `train_config.base_channels` matching the consuming Table-8 template
    (`256` for ablation-default, `640` for B/2 and L/2),
  - explicit training envelope fields (`steps`, `batch_size`, `learning_rate`, scheduler/warmup)
    so artifacts can be tagged as `bootstrap/closest-feasible` vs `paper-scale`,
  - masking ratio / patch settings,
  - optimizer + schedule fields,
  - checkpoint/export paths.
- encoder export payload (`mae_encoder.pt`) containing:
  - `config` metadata (`in_channels`, `base_channels`, `stages`, `encoder_arch`),
  - `encoder_state_dict`.
- environment artifacts:
  - `env_snapshot.json`,
  - `codebase_fingerprint.json`,
  - `env_fingerprint.json`.
- `RUN.md` with command provenance and key paths.

## 2) Latent faithful-template rerun evidence (minimal set)

For each reportable faithful-template run, require all of:

- train run artifacts:
  - `latent_summary.json`,
  - resolved config path + hash,
  - checkpoint directory with periodic checkpoints and final checkpoint.
- evaluation artifacts:
  - decoded sample manifest (`sample_summary.json`) for the evaluated checkpoint,
  - `eval_pretrained.json` (pretrained Inception only),
  - reference stats path and contract hash/provenance.
- stability/audit artifacts:
  - last-K checkpoint evaluation summary,
  - alpha sweep summary on a fixed checkpoint,
  - nearest-neighbor audit JSON.
- reproducibility artifacts:
  - `RUN.md`,
  - env/codebase fingerprints.

## 3) Release-gate link

`docs/release_gate_checklist.md` is the policy checklist; this file defines the concrete artifact payload required to satisfy its faithfulness evidence gates.

## 4) Provenance contract link

Dataset/tokenizer/evaluator provenance fields are defined in:
- `docs/provenance_contract.md`
