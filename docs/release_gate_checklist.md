# Release Gate Checklist (Parity-Critical)

This checklist defines what must be true before any paper-faithfulness claim is promoted.

## 1) Mandatory deterministic tests

Run and pass all:

- `uv run pytest -q tests/unit/test_queue_label_integrity.py`
- `uv run pytest -q tests/unit/test_feature_drift_loss.py`
- `uv run pytest -q tests/unit/test_table8_faithful_config_contract.py`
- `uv run pytest -q tests/unit/test_provenance_contract.py`
- `uv run pytest -q tests/unit/test_compile_toggle_helpers.py`
- `uv run pytest -q tests/integration/test_eval_fid_is_cache_roundtrip.py`
- `uv run pytest -q tests/integration/test_cache_reference_stats_smoke.py`
- `uv run python scripts/check_queue_determinism.py --device cpu --output-path outputs/stage4_queue_determinism/queue_determinism.json`
- `uv run python scripts/check_feature_loss_reduction_scaling.py --output-path outputs/stage4_reduction_scaling/reduction_scaling_check.json`
- `uv run python scripts/audit_claim_eval_contract.py --eval-json <CLAIM_EVAL_1> --output-json <OUT_JSON>`

## 2) Mandatory artifact bundle per run

For each reportable run, require:

- train summary (`latent_summary.json` or `pixel_summary.json`)
- resolved config hash + config path
- `RUN.md`
- environment snapshot + fingerprints
- checkpoint path(s)
- eval summary (`eval_pretrained.json` or equivalent)
- reference stats path + contract hash
- post-run rung summary (if part of D2 ladder)

For paper-facing faithfulness claims, require the full bundle in:
- `docs/faithfulness_evidence_requirements.md`

## 3) Comparability confidence tiers

- **Tier A (paper-facing comparable)**:
  - pretrained Inception,
  - contract-matching reference stats,
  - fixed protocol + provenance hashes,
  - no unresolved parity-critical checklist items.
- **Tier B (repo-comparable internal)**:
  - contract-matching protocol but reduced sample count and/or proxy horizons.
- **Tier C (pipeline/CI only)**:
  - random-weight inception (`--inception-weights none`) or smoke-only settings.

Only Tier A can support paper-faithfulness claims.
Tier C (smoke/CI artifacts, including `inception_weights=none`) must be excluded from claim-facing summaries.

## 4) Merge gate checklist (parity-critical changes)

For any PR touching queue/drift/eval semantics:

- [ ] Add/adjust tests for changed semantics.
- [ ] Update `docs/decision_log.md` with rationale and residual risk.
- [ ] Update `docs/reproduction_report.md` wording if claim scope changes.
- [ ] Update `docs/eval_contract.md` if eval protocol changed.
- [ ] Update `docs/claim_to_evidence_matrix.md` claim mappings.
- [ ] Update `docs/deviations_table.md` deviation impact entries.
- [ ] Provide benchmark/parity artifacts for performance-path changes.
