from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    args = _parse_args()
    summary_path = Path(args.summary_path)
    output_path = Path(args.output_path)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    logs = list(payload.get("logs", []))
    if not logs:
        raise ValueError("No logs found in latent summary")

    losses = [float(entry["loss"]) for entry in logs]
    drift_norms = [float(entry["mean_drift_norm"]) for entry in logs]
    finite = all(math.isfinite(value) for value in losses + drift_norms)

    min_loss = float(min(losses))
    final_loss = float(losses[-1])
    first_loss = float(losses[0])
    final_drift = float(drift_norms[-1])
    first_drift = float(drift_norms[0])
    overfit_flag = bool(payload.get("train_config", {}).get("overfit_fixed_batch", False))

    passed = (
        finite
        and overfit_flag
        and min_loss <= float(args.loss_threshold)
        and final_loss <= float(args.loss_threshold)
    )
    report = {
        "summary_path": str(summary_path),
        "overfit_fixed_batch": overfit_flag,
        "finite_metrics": finite,
        "loss_threshold": float(args.loss_threshold),
        "first_loss": first_loss,
        "min_loss": min_loss,
        "final_loss": final_loss,
        "first_mean_drift_norm": first_drift,
        "final_mean_drift_norm": final_drift,
        "passed": passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check tiny-set latent overfit sanity signal.")
    parser.add_argument("--summary-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--loss-threshold", type=float, default=1e-6)
    return parser.parse_args()


if __name__ == "__main__":
    main()
