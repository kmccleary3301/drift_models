# Deviations Table (Current)

This table tracks known deviations from strict paper-faithful execution and their impact.

| Area | Current state | Impact | Mitigation / gate |
| --- | --- | --- | --- |
| Long-horizon active latent run config | Running from `closest-feasible` config lineage (`outputs/imagenet/paperscale_b2_corrected_restart_nokernelcompile_20260219_152045/config.yaml`) | Not eligible as faithful-template evidence | Keep tier as `closest-feasible`; do not promote to faithful claim path |
| Full faithful-template latent horizon | Pending completion (`G3.1..G3.6`) | Blocks final paper-facing metric claim closure | Complete faithful run + eval package with strict queue and pretrained eval gates |
| Pixel paper-faithful path | Pre-production package completed (`outputs/feature_ablations/pixel_paper_facing_package_preprod_convnext_20260220`), but full paper-scale parity/evidence still pending (`G4.*`) | Blocks pixel paper-faithful claims | Keep pixel scope experimental/closest-feasible until paper-scale package + remaining parity gates close |
| Open/ambiguous paper details (alpha embedding/RoPE-QK micro-variants) | Deferred with impact (`docs/decision_closure_status.md`) | Limits strength of strict architecture-level parity claims | Track as deferred-with-impact; close with targeted ablations/docs updates |
| Older eval artifacts missing modern provenance fields | Some historic eval JSON files lack full `inception_provenance` payload | Those artifacts should not be used for final claim package | Use claim-eval audit gate and prefer newer contract-compliant eval artifacts |
