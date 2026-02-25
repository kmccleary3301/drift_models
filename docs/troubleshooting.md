# Troubleshooting

## `RuntimeError: No CUDA GPUs are available`

- Check GPU visibility:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
```

- Ensure you are not masking devices via `CUDA_VISIBLE_DEVICES`.
- Use `--device cpu` when GPU is unavailable.
- If you used `--device gpu`, switch to explicit `--device cpu` or check accelerator visibility first.

## `Requested --device ... backend ... unavailable`

- Run with `--device auto` to let the runtime pick a supported backend.
- For deterministic runs, pin to an available explicit device index (`cuda:0`, `cuda:1`, `xpu:0`, or `cpu`).

## `torch.compile` instability

- Use explicit fail-action policy:
  - `--compile-fail-action warn`: fallback to eager + warning
  - `--compile-fail-action raise`: hard fail for strict lanes
  - `--compile-fail-action disable`: skip compile attempt entirely
- For MPS/XPU, prefer `--compile-fail-action disable`.
- Disable compile flags when needed:
  - remove `--compile-generator`
  - remove `--feature-compile-drift-kernel`

## `compile.smoke(...)` skipped in runtime preflight

- This is expected when compile policy is `disable` or backend is policy-unsupported.
- Re-run preflight with strict compile failure behavior only when intentionally validating compile:

```bash
uv run python scripts/runtime_preflight.py \
  --device cpu \
  --check-compile \
  --compile-fail-action raise \
  --strict
```

## MPS missing op / NotImplemented

- Set:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

- Retry on CPU if behavior is unstable.

## ROCm / XPU backend mismatch

- Confirm backend-specific wheel/runtime installation first.
- Run preflight before training:

```bash
uv run python scripts/runtime_preflight.py --device auto --check-compile --compile-fail-action disable --strict
```

- Treat backend runs as best-effort unless listed first-class in `docs/compatibility_matrix.md`.

## Queue underflow / label backfill errors

- Verify real-batch source points at valid datasets/manifests.
- Increase queue warmup and coverage settings.
- Check class coverage in queue reports (`queue_covered_classes`).

## FID/IS cache contract mismatch

- Ensure reference cache matches eval contract.
- Only use `--allow-reference-contract-mismatch` intentionally for legacy cache comparisons.
