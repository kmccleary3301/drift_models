# Hardware What-If: 2× H200 vs Current Single-GPU Lane

## Context

- Current long-horizon lane: single-GPU training (`outputs/imagenet/paperscale_b2_closest_feasible_run_20260215_005727`).
- Observed throughput in this lane is constrained by a mix of model compute, queue/data orchestration, and checkpoint/eval overhead.
- Planner critique guidance (`docs_tmp/critique/P2_CRITIQUE_PLANNER_RESPONSE.md`) estimates practical scaling below ideal linear speedup.

## Practical expectation

- **Realistic end-to-end speedup** on 2× H200 with clean DDP and current code structure: **~1.7× to ~1.9×**.
- Not 2.0× because fixed overheads remain:
  - data/queue orchestration and host-side work,
  - checkpoint write time,
  - evaluation cadence and non-overlapped sampling/eval,
  - distributed synchronization costs.

## Preconditions to approach upper bound

1. Keep vectorized drift kernels (already merged) and avoid Python-slot loops.
2. Use deterministic DDP setup with explicit rank-local seed control.
3. Keep compile scope narrow (`generator.forward` and vetted drift kernels only).
4. Move eval/sampling off the primary training lane or schedule sparsely.
5. Keep storage throughput high enough to avoid checkpoint stalls.

## Decision use

- Treat 2× H200 as a **time compression lever**, not a semantic correctness substitute.
- Correctness/fidelity gates (P0 + rung ladder + release gates) remain unchanged.
