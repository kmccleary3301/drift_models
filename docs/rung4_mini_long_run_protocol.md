# D2 Rung4 Mini-Long Run Protocol

## Objective

Run a corrected-semantics mini-long horizon (`~10k` steps) and attach scheduled proxy eval artifacts.

## Train command

```bash
uv run python scripts/train_latent.py \
  --config configs/latent/rung4_mini_long_protocol.yaml \
  --output-dir outputs/stage4_mini_long_run \
  --checkpoint-dir outputs/stage4_mini_long_run/checkpoints
```

## Scheduled proxy eval checkpoints

Evaluate immutable checkpoints at steps `2000, 5000, 10000`:

```bash
scripts/run_rung4_proxy_eval.sh outputs/stage4_mini_long_run cuda:0
```

## Pass/fail gate (rung4)

- pass if:
  - run reaches step `10000` without NaN/inf,
  - proxy eval completes on the targeted checkpoint set,
  - summary + checkpoint artifacts are present.
- fail otherwise.
