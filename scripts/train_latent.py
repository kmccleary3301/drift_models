from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn

from drifting_models.data import (
    ClassConditionalSampleQueue,
    GroupedSamplingConfig,
    QueueConfig,
    RealBatchProvider,
    RealBatchProviderConfig,
    sample_grouped_real_batches,
)
from drifting_models.drift_field import DriftFieldConfig
from drifting_models.drift_field import cfg_alpha_to_unconditional_weight
from drifting_models.utils.alpha import sample_alpha
from drifting_models.drift_loss import DriftingLossConfig, FeatureDriftingConfig
from drifting_models.features import (
    FeatureVectorizationConfig,
    LatentResNetMAE,
    LatentResNetMAEConfig,
    LatentDecoderConfig,
    TinyFeatureEncoder,
    TinyFeatureEncoderConfig,
    build_latent_decoder,
)
from drifting_models.models import DiTLikeConfig, DiTLikeGenerator
from drifting_models.train import GroupedDriftStepConfig, grouped_drift_training_step
from drifting_models.utils import (
    add_device_argument,
    ModelEMA,
    codebase_fingerprint,
    environment_fingerprint,
    environment_snapshot,
    file_sha256,
    load_training_checkpoint,
    maybe_compile_callable,
    payload_sha256,
    resolve_device,
    save_training_checkpoint,
    seed_everything,
    write_json,
)
from drifting_models.utils.run_md import write_run_md


def main() -> None:
    args = _parse_args()
    args = _apply_config_overrides(args)
    _assert_single_process()
    if args.style_token_count < 0:
        raise ValueError("--style-token-count must be >= 0")
    if args.style_token_count > 0 and args.style_vocab_size <= 0:
        raise ValueError("--style-vocab-size must be > 0 when --style-token-count > 0")
    if args.overfit_fixed_batch and args.use_queue:
        raise ValueError("--overfit-fixed-batch is incompatible with --use-queue")
    config_hash = None if args.config is None else file_sha256(Path(args.config))
    resolved_config_hash = payload_sha256(_resume_fingerprint_payload(args))
    env_fingerprint = environment_fingerprint()
    repo_root = Path(__file__).resolve().parents[1]
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "env_snapshot.json", environment_snapshot(paths=[output_dir]))
        write_json(output_dir / "codebase_fingerprint.json", codebase_fingerprint(repo_root=repo_root))
    seed_everything(args.seed)
    device = resolve_device(args.device)

    model_config = DiTLikeConfig(
        image_size=args.image_size,
        in_channels=args.channels,
        out_channels=args.channels,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        ffn_inner_dim=args.ffn_inner_dim,
        num_classes=args.num_classes,
        register_tokens=args.register_tokens,
        style_vocab_size=args.style_vocab_size,
        style_token_count=args.style_token_count,
        norm_type=args.norm_type,
        use_qk_norm=args.use_qk_norm,
        use_rope=args.use_rope,
    )
    model = DiTLikeGenerator(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        weight_decay=float(args.weight_decay),
    )
    start_step = 0
    resume_payload = None
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        resume_probe = torch.load(resume_path, map_location=device)
        resume_extra = resume_probe.get("extra", {})
        if isinstance(resume_extra, dict):
            resume_resolved_hash = resume_extra.get("resolved_config_hash")
            if (
                isinstance(resume_resolved_hash, str)
                and resume_resolved_hash != resolved_config_hash
                and not args.allow_resume_config_mismatch
            ):
                raise ValueError(
                    "Resume config mismatch detected. "
                    "Use --allow-resume-config-mismatch to override."
                )
        if args.resume_model_only:
            model.load_state_dict(resume_probe["model_state_dict"])
            resume_payload = resume_probe
        else:
            resume_payload = load_training_checkpoint(
                path=resume_path,
                model=model,
                optimizer=optimizer,
                map_location=device,
            )
        start_step = int(resume_payload.get("step", 0))
        if not args.resume_model_only and args.resume_reset_optimizer_lr:
            for group in optimizer.param_groups:
                group["lr"] = float(args.learning_rate)
                group["initial_lr"] = float(args.learning_rate)
    compile_info = _maybe_compile_generator_forward(model=model, args=args)
    queue = None
    sampling_config = None
    real_batch_provider = None
    real_batch_manifest_fingerprint = None
    real_provider_sanity_report = None
    queue_warmup_report = None
    underflow_missing_labels = 0
    underflow_backfilled_samples = 0
    queue_resumed_from_checkpoint = False
    if args.use_queue:
        if args.unconditional_per_group <= 0:
            raise ValueError("--unconditional-per-group must be > 0 when --use-queue is enabled")
        if args.queue_strict_without_replacement:
            if args.queue_per_class_capacity < args.positives_per_group:
                raise ValueError(
                    "--queue-per-class-capacity must be >= --positives-per-group when "
                    "--queue-strict-without-replacement is enabled"
                )
            if args.queue_global_capacity < args.unconditional_per_group:
                raise ValueError(
                    "--queue-global-capacity must be >= --unconditional-per-group when "
                    "--queue-strict-without-replacement is enabled"
                )
        real_provider_config = RealBatchProviderConfig(
            source=args.real_batch_source,
            dataset_size=args.real_dataset_size,
            batch_size=args.real_loader_batch_size,
            shuffle=not args.disable_real_shuffle,
            num_workers=int(args.real_num_workers),
            pin_memory=bool(args.real_pin_memory),
            persistent_workers=bool(args.real_persistent_workers),
            prefetch_factor=None if args.real_prefetch_factor <= 0 else int(args.real_prefetch_factor),
            seed=args.seed,
            channels=args.channels,
            image_size=args.image_size,
            num_classes=args.num_classes,
            imagefolder_root=args.real_imagefolder_root,
            webdataset_urls=args.real_webdataset_urls,
            tensor_file_path=args.real_tensor_file_path,
            tensor_shards_manifest_path=args.real_tensor_shards_manifest_path,
            transform_resize=args.real_transform_resize,
            transform_center_crop=not args.disable_real_center_crop,
            transform_horizontal_flip=args.real_horizontal_flip,
            transform_normalize=args.real_transform_normalize,
        )
        if args.real_sanity_sample_batches > 0:
            real_sanity_provider = RealBatchProvider(
                replace(
                    real_provider_config,
                    seed=int(real_provider_config.seed + 9999),
                    shuffle=True,
                )
            )
            real_provider_sanity_report = _build_real_provider_sanity_report(
                provider=real_sanity_provider,
                sample_batches=int(args.real_sanity_sample_batches),
                num_classes=int(args.num_classes),
                device=device,
            )
        real_batch_provider = RealBatchProvider(real_provider_config)
        real_batch_manifest_fingerprint = real_batch_provider.manifest_fingerprint
        queue = ClassConditionalSampleQueue(
            QueueConfig(
                num_classes=args.num_classes,
                per_class_capacity=args.queue_per_class_capacity,
                global_capacity=args.queue_global_capacity,
                store_device=args.queue_store_device,
                strict_without_replacement=bool(args.queue_strict_without_replacement),
            )
        )
        sampling_config = GroupedSamplingConfig(
            positives_per_group=args.positives_per_group,
            unconditional_per_group=args.unconditional_per_group,
        )
        if resume_payload is not None and isinstance(resume_payload.get("queue_state"), dict):
            resume_queue_state = resume_payload["queue_state"]
            try:
                queue.load_state_dict(resume_queue_state)
                queue_resumed_from_checkpoint = True
            except ValueError:
                if not args.allow_resume_config_mismatch:
                    raise
        if queue_resumed_from_checkpoint:
            resume_covered_classes = _queue_covered_class_count(queue)
            if resume_covered_classes < int(args.num_classes):
                _prime_queue(
                    queue=queue,
                    provider=real_batch_provider,
                    num_classes=args.num_classes,
                    sample_count=args.queue_prime_samples,
                    warmup_mode=args.queue_warmup_mode,
                    warmup_min_per_class=args.queue_warmup_min_per_class,
                    device=device,
                )
                queue_warmup_report = _build_queue_warmup_report(
                    queue=queue,
                    warmup_mode=f"resume_reprime_{args.queue_warmup_mode}",
                    samples_pushed=args.queue_prime_samples,
                    report_level=str(args.queue_report_level),
                )
                queue_warmup_report["resume_initial_covered_classes"] = float(resume_covered_classes)
            else:
                queue_warmup_report = {
                    "mode": "resume_restore",
                    "samples_pushed": 0.0,
                    "report_level": str(args.queue_report_level),
                    "min_count": float(min(queue.class_counts(), default=0)),
                    "max_count": float(max(queue.class_counts(), default=0)),
                    "global_count": float(queue.global_count()),
                    "covered_classes": float(resume_covered_classes),
                }
        else:
            _prime_queue(
                queue=queue,
                provider=real_batch_provider,
                num_classes=args.num_classes,
                sample_count=args.queue_prime_samples,
                warmup_mode=args.queue_warmup_mode,
                warmup_min_per_class=args.queue_warmup_min_per_class,
                device=device,
            )
            queue_warmup_report = _build_queue_warmup_report(
                queue=queue,
                warmup_mode=args.queue_warmup_mode,
                samples_pushed=args.queue_prime_samples,
                report_level=str(args.queue_report_level),
            )
    feature_extractor = None
    feature_config = None
    feature_input_transform = None
    if args.use_feature_loss:
        feature_channels = args.channels
        if args.latent_feature_decode_mode != "none":
            decoder_mode = "identity" if args.latent_feature_decode_mode == "identity" else "conv"
            decoder = build_latent_decoder(
                LatentDecoderConfig(
                    mode=decoder_mode,
                    latent_channels=args.channels,
                    out_channels=args.latent_decoder_out_channels,
                    image_size=args.latent_decoder_image_size,
                    hidden_channels=args.latent_decoder_hidden_channels,
                )
            ).to(device)
            for parameter in decoder.parameters():
                parameter.requires_grad = False
            feature_input_transform = decoder
            if decoder_mode == "conv":
                feature_channels = args.latent_decoder_out_channels
        feature_extractor = _build_feature_extractor(
            args=args,
            device=device,
            in_channels=int(feature_channels),
        )
        feature_extractor.eval()
        for parameter in feature_extractor.parameters():
            parameter.requires_grad = False
        feature_config = FeatureDriftingConfig(
            temperatures=tuple(args.feature_temperatures),
            vectorization=FeatureVectorizationConfig(
                include_per_location=not args.disable_per_location,
                include_global_stats=not args.disable_global_stats,
                include_patch2_stats=not args.disable_patch2_stats,
                include_patch4_stats=args.include_patch4_stats,
                include_input_x2_mean=args.include_input_x2_mean,
                selected_stages=None
                if len(args.feature_selected_stages) == 0
                else tuple(args.feature_selected_stages),
            ),
            normalize_features=True,
            normalize_drifts=True,
            temperature_aggregation=str(args.feature_temperature_aggregation),
            loss_term_reduction=str(args.feature_loss_term_reduction),
            scale_temperature_by_sqrt_channels=not args.disable_feature_temperature_sqrt_scaling,
            detach_positive_features=True,
            detach_negative_features=True,
            share_location_normalization=not args.disable_shared_location_normalization,
            include_raw_drift_loss=bool(args.feature_include_raw_drift_loss),
            raw_drift_loss_weight=float(args.feature_raw_drift_loss_weight),
            compile_drift_kernel=bool(args.feature_compile_drift_kernel),
            compile_drift_backend=str(args.feature_compile_backend),
            compile_drift_mode=str(args.feature_compile_mode),
            compile_drift_dynamic=bool(args.feature_compile_dynamic),
            compile_drift_fullgraph=bool(args.feature_compile_fullgraph),
            compile_drift_fail_action=str(args.feature_compile_fail_action),
        )

    step_config = GroupedDriftStepConfig(
        loss_config=DriftingLossConfig(
            drift_field=DriftFieldConfig(
                temperature=args.temperature,
                normalize_over_x=True,
                mask_self_negatives=True,
            ),
            attraction_scale=1.0,
            repulsion_scale=1.0,
            stopgrad_target=True,
        ),
        feature_config=feature_config,
        drift_temperatures=tuple(float(value) for value in args.drift_temperatures),
        drift_temperature_reduction=str(args.drift_temperature_reduction),
        clip_grad_norm=2.0,
        run_optimizer_step=True,
    )

    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be > 0")
    amp_dtype = None
    grad_scaler = None
    if device.type == "cuda":
        if args.precision == "bf16":
            amp_dtype = torch.bfloat16
        elif args.precision == "fp16":
            amp_dtype = torch.float16
            grad_scaler = torch.amp.GradScaler("cuda")
    lr_scheduler = _build_lr_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
    )
    if resume_payload is not None and lr_scheduler is not None and args.resume_reset_scheduler:
        lr_scheduler.last_epoch = int(start_step) - 1
    if resume_payload is not None and not args.resume_model_only and not args.resume_reset_scheduler:
        if lr_scheduler is not None and "scheduler_state_dict" in resume_payload:
            lr_scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if grad_scaler is not None and "scaler_state_dict" in resume_payload:
            grad_scaler.load_state_dict(resume_payload["scaler_state_dict"])
    ema_model = ModelEMA.create(model=model, decay=args.ema_decay) if args.use_ema else None
    if ema_model is not None and resume_payload is not None and "ema_state_dict" in resume_payload:
        ema_model.load_state_dict(resume_payload["ema_state_dict"])

    overfit_class_labels = None
    overfit_alpha = None
    overfit_noise_grouped = None
    overfit_positives_grouped = None
    overfit_style_indices = None
    if args.overfit_fixed_batch:
        overfit_class_labels = torch.randint(0, args.num_classes, (args.groups,), device=device)
        overfit_alpha = _sample_alpha(args, device=device)
        overfit_noise_grouped = torch.randn(
            args.groups,
            args.negatives_per_group,
            args.channels,
            args.image_size,
            args.image_size,
            device=device,
        )
        overfit_positives_grouped = _sample_synthetic_positives(
            groups=args.groups,
            positives_per_group=args.positives_per_group,
            channels=args.channels,
            image_size=args.image_size,
            class_labels=overfit_class_labels,
            num_classes=args.num_classes,
            device=device,
        )
        if model_config.style_token_count > 0:
            overfit_style_indices = torch.randint(
                0,
                model_config.style_vocab_size,
                (args.groups, args.negatives_per_group, model_config.style_token_count),
                device=device,
            )

    logs: list[dict[str, float]] = []
    loop_start_time = time.perf_counter()
    for step in range(start_step, args.steps):
        step_start_time = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if args.grad_accum_steps > 1:
            optimizer.zero_grad(set_to_none=True)
        micro_stats: list[dict[str, float]] = []
        for accum_index in range(args.grad_accum_steps):
            if args.overfit_fixed_batch:
                if (
                    overfit_class_labels is None
                    or overfit_alpha is None
                    or overfit_noise_grouped is None
                    or overfit_positives_grouped is None
                ):
                    raise RuntimeError("overfit cache is not initialized")
                class_labels = overfit_class_labels
                alpha = overfit_alpha
                noise_grouped = overfit_noise_grouped
                positives_grouped = overfit_positives_grouped
                style_indices = overfit_style_indices
            else:
                class_labels = torch.randint(0, args.num_classes, (args.groups,), device=device)
                alpha = _sample_alpha(args, device=device)
                noise_grouped = torch.randn(
                    args.groups,
                    args.negatives_per_group,
                    args.channels,
                    args.image_size,
                    args.image_size,
                    device=device,
                )
                positives_grouped = _sample_synthetic_positives(
                    groups=args.groups,
                    positives_per_group=args.positives_per_group,
                    channels=args.channels,
                    image_size=args.image_size,
                    class_labels=class_labels,
                    num_classes=args.num_classes,
                    device=device,
                )
                style_indices = None
                if model_config.style_token_count > 0:
                    style_indices = torch.randint(
                        0,
                        model_config.style_vocab_size,
                        (args.groups, args.negatives_per_group, model_config.style_token_count),
                        device=device,
                    )
            unconditional_grouped = None
            unconditional_weight_grouped = None
            if args.use_queue and queue is not None and sampling_config is not None:
                if real_batch_provider is None:
                    raise RuntimeError("real_batch_provider must be initialized when queue mode is enabled")
                should_refill = args.queue_refill_policy == "per_step" or (
                    args.queue_refill_policy == "every_n_steps"
                    and ((step - start_step) % max(1, args.queue_refill_every) == 0)
                )
                if should_refill:
                    new_real_images, new_real_labels = _sample_real_batch(
                        provider=real_batch_provider,
                        count=args.queue_push_batch,
                        device=device,
                    )
                    queue.push(new_real_images, new_real_labels)
                backfilled = _ensure_queue_has_labels(
                    queue=queue,
                    class_labels=class_labels,
                    provider=real_batch_provider,
                    required_count=(
                        int(args.positives_per_group) if args.queue_strict_without_replacement else 1
                    ),
                    device=device,
                )
                underflow_missing_labels += backfilled
                underflow_backfilled_samples += backfilled
                positives_grouped, unconditional_grouped = sample_grouped_real_batches(
                    queue=queue,
                    class_labels=class_labels,
                    config=sampling_config,
                    device=device,
                )
                unconditional_weight_grouped = torch.tensor(
                    [
                        cfg_alpha_to_unconditional_weight(
                            alpha=float(alpha[g].item()),
                            n_generated_negatives=args.negatives_per_group,
                            n_unconditional_negatives=args.unconditional_per_group,
                        )
                        for g in range(args.groups)
                    ],
                    device=device,
                    dtype=torch.float32,
                )

            micro_run_optimizer_step = accum_index == (args.grad_accum_steps - 1)
            micro_config = replace(step_config, run_optimizer_step=micro_run_optimizer_step)
            micro_stats.append(
                grouped_drift_training_step(
                    generator=model,
                    optimizer=optimizer,
                    noise_grouped=noise_grouped,
                    class_labels_grouped=class_labels,
                    alpha_grouped=alpha,
                    positives_grouped=positives_grouped,
                    style_indices_grouped=style_indices,
                    unconditional_grouped=unconditional_grouped,
                    unconditional_weight_grouped=unconditional_weight_grouped,
                    feature_extractor=feature_extractor,
                    feature_input_transform=feature_input_transform,
                    amp_dtype=amp_dtype,
                    grad_scaler=grad_scaler,
                    backward_when_no_step=args.grad_accum_steps > 1 and not micro_run_optimizer_step,
                    loss_divisor=float(args.grad_accum_steps),
                    zero_grad_before_backward=args.grad_accum_steps == 1,
                    config=micro_config,
                )
            )
        stats = _mean_numeric_stats(micro_stats)
        _attach_loss_scale_metrics(stats)
        stats["lr"] = float(optimizer.param_groups[0]["lr"])
        if ema_model is not None:
            ema_model.update(model)
        if lr_scheduler is not None:
            lr_scheduler.step()
        step_time_s = time.perf_counter() - step_start_time
        generated_count = float(args.groups * args.negatives_per_group)
        stats["step_time_s"] = float(step_time_s)
        stats["generated_images_per_sec"] = float(generated_count / max(step_time_s, 1e-8))
        stats["peak_cuda_mem_mb"] = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        )
        if args.use_queue:
            stats["queue_underflow_backfilled"] = float(underflow_backfilled_samples)
            stats["queue_underflow_missing_labels"] = float(underflow_missing_labels)
            if queue is not None:
                counts = queue.class_counts()
                stats["queue_global_count"] = float(queue.global_count())
                stats["queue_covered_classes"] = float(sum(1 for count in counts if count > 0))
        if step == start_step or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            logs.append({"step": float(step + 1), **stats})
            _emit_progress_line(
                step=step + 1,
                total_steps=args.steps,
                start_step=start_step,
                loop_start_time=loop_start_time,
                stats=stats,
            )
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            _save_checkpoints(
                args=args,
                model=model,
                optimizer=optimizer,
                queue=queue,
                step=step + 1,
                extra={
                    "config_path": args.config,
                    "config_hash": config_hash,
                    "resolved_config_hash": resolved_config_hash,
                    "model_config": asdict(model_config),
                },
                scheduler=lr_scheduler,
                scaler=grad_scaler,
                ema_state_dict=None if ema_model is None else ema_model.state_dict(),
            )

    _save_checkpoints(
        args=args,
        model=model,
        optimizer=optimizer,
        queue=queue,
        step=args.steps,
        extra={
            "config_path": args.config,
            "config_hash": config_hash,
            "resolved_config_hash": resolved_config_hash,
            "model_config": asdict(model_config),
        },
        scheduler=lr_scheduler,
        scaler=grad_scaler,
        ema_state_dict=None if ema_model is None else ema_model.state_dict(),
        force=bool(args.checkpoint_path is not None or args.checkpoint_dir is not None),
    )

    summary = {
        "device": str(device),
        "seed": args.seed,
        "config_path": args.config,
        "config_hash": config_hash,
        "resolved_config_hash": resolved_config_hash,
        "resume_from": args.resume_from,
        "resume_step": float(start_step),
        "env_fingerprint": env_fingerprint,
        "model_config": asdict(model_config),
        "train_config": {
            "steps": args.steps,
            "groups": args.groups,
            "negatives_per_group": args.negatives_per_group,
            "positives_per_group": args.positives_per_group,
            "alpha_fixed": None if args.alpha_fixed is None else float(args.alpha_fixed),
            "alpha_min": float(args.alpha_min),
            "alpha_max": float(args.alpha_max),
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "adam_beta1": float(args.adam_beta1),
            "adam_beta2": float(args.adam_beta2),
            "weight_decay": float(args.weight_decay),
            "precision": args.precision,
            "grad_accum_steps": args.grad_accum_steps,
            "scheduler": args.scheduler,
            "warmup_steps": args.warmup_steps,
            "use_ema": bool(args.use_ema),
            "ema_decay": args.ema_decay,
            "use_feature_loss": bool(args.use_feature_loss),
            "feature_encoder": args.feature_encoder,
            "mae_encoder_path": args.mae_encoder_path,
            "mae_encoder_arch": args.mae_encoder_arch,
            "use_queue": bool(args.use_queue),
            "queue_resumed_from_checkpoint": bool(queue_resumed_from_checkpoint),
            "latent_feature_decode_mode": args.latent_feature_decode_mode,
            "checkpoint_path": args.checkpoint_path,
            "checkpoint_dir": args.checkpoint_dir,
            "keep_last_k_checkpoints": args.keep_last_k_checkpoints,
            "save_every": args.save_every,
            "queue_warmup_mode": args.queue_warmup_mode,
            "queue_warmup_min_per_class": args.queue_warmup_min_per_class,
            "queue_refill_policy": args.queue_refill_policy,
            "queue_refill_every": args.queue_refill_every,
            "queue_strict_without_replacement": bool(args.queue_strict_without_replacement),
            "clip_grad_norm": 2.0,
            "overfit_fixed_batch": bool(args.overfit_fixed_batch),
            "resume_model_only": bool(args.resume_model_only),
            "resume_reset_scheduler": bool(args.resume_reset_scheduler),
            "resume_reset_optimizer_lr": bool(args.resume_reset_optimizer_lr),
            "compile_generator": bool(args.compile_generator),
            "compile_backend": str(args.compile_backend),
            "compile_mode": str(args.compile_mode),
            "compile_dynamic": bool(args.compile_dynamic),
            "compile_fullgraph": bool(args.compile_fullgraph),
            "compile_fail_action": str(args.compile_fail_action),
        },
        "compile": compile_info,
        "logs": logs,
    }
    if logs and all("alpha_mean" in entry and "alpha_min" in entry and "alpha_max" in entry for entry in logs):
        summary["alpha_stats"] = {
            "mean_over_steps": float(sum(entry["alpha_mean"] for entry in logs) / len(logs)),
            "min_over_steps": float(min(entry["alpha_min"] for entry in logs)),
            "max_over_steps": float(max(entry["alpha_max"] for entry in logs)),
        }
    if real_batch_provider is not None:
        summary["real_batch_provider"] = {
            "source": real_batch_provider.config.source,
            "manifest_fingerprint": real_batch_manifest_fingerprint,
            "config": real_batch_provider.config.__dict__,
        }
    if queue_warmup_report is not None:
        summary["queue_warmup_report"] = queue_warmup_report
    if args.use_queue:
        summary["queue_underflow_totals"] = {
            "missing_labels": float(underflow_missing_labels),
            "backfilled_samples": float(underflow_backfilled_samples),
        }
        summary["queue_report_level"] = str(args.queue_report_level)
    if real_provider_sanity_report is not None:
        summary["real_provider_sanity_report"] = real_provider_sanity_report
    if logs:
        summary["perf"] = {
            "mean_step_time_s": float(sum(entry["step_time_s"] for entry in logs) / len(logs)),
            "mean_generated_images_per_sec": float(
                sum(entry["generated_images_per_sec"] for entry in logs) / len(logs)
            ),
            "max_peak_cuda_mem_mb": float(max(entry["peak_cuda_mem_mb"] for entry in logs)),
        }
    if logs and all("loss_per_term" in entry for entry in logs):
        summary["loss_scale"] = {
            "last_loss": float(logs[-1]["loss"]),
            "last_loss_term_count": float(logs[-1]["loss_term_count"]),
            "last_loss_per_term": float(logs[-1]["loss_per_term"]),
            "mean_loss_per_term": float(sum(entry["loss_per_term"] for entry in logs) / len(logs)),
        }
        if all("feature_loss_per_term" in entry for entry in logs):
            summary["loss_scale"]["mean_feature_loss_per_term"] = float(
                sum(entry["feature_loss_per_term"] for entry in logs) / len(logs)
            )
    if resume_payload is not None and "extra" in resume_payload:
        summary["resume_extra"] = resume_payload["extra"]
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        write_json(output_dir / "latent_summary.json", summary)
        write_json(output_dir / "env_fingerprint.json", env_fingerprint)
        write_run_md(
            output_dir / "RUN.md",
            {
                "output_root": str(output_dir),
                "paths": {
                    "latent_summary_json": str(output_dir / "latent_summary.json"),
                    "env_snapshot_json": str(output_dir / "env_snapshot.json"),
                    "codebase_fingerprint_json": str(output_dir / "codebase_fingerprint.json"),
                    "env_fingerprint_json": str(output_dir / "env_fingerprint.json"),
                    "checkpoint_path": args.checkpoint_path,
                    "checkpoint_dir": args.checkpoint_dir,
                },
                "args": vars(args),
                "commands": {"train": {"argv": list(sys.argv), "returncode": 0}},
            },
        )
    print(json.dumps(summary, indent=2))


def _emit_progress_line(
    *,
    step: int,
    total_steps: int,
    start_step: int,
    loop_start_time: float,
    stats: dict[str, float],
) -> None:
    done = max(1, step - start_step)
    elapsed = time.perf_counter() - loop_start_time
    mean_step = elapsed / float(done)
    remaining = max(0, total_steps - step)
    eta_s = mean_step * float(remaining)

    lr = stats.get("lr")
    loss = stats.get("loss")
    step_time = stats.get("step_time_s")
    img_s = stats.get("generated_images_per_sec")
    peak = stats.get("peak_cuda_mem_mb")
    loss_term_count = stats.get("loss_term_count")
    loss_per_term = stats.get("loss_per_term")

    parts = [
        f"[train_latent] step={step}/{total_steps}",
        f"lr={lr:.3e}" if isinstance(lr, (int, float)) else None,
        f"loss={loss:.4f}" if isinstance(loss, (int, float)) else None,
        f"loss/term={loss_per_term:.4f}" if isinstance(loss_per_term, (int, float)) else None,
        (
            f"terms={int(round(loss_term_count))}"
            if isinstance(loss_term_count, (int, float)) and loss_term_count > 0
            else None
        ),
        f"dt={step_time:.3f}s" if isinstance(step_time, (int, float)) else None,
        f"img/s={img_s:.1f}" if isinstance(img_s, (int, float)) else None,
        f"peak={peak:.0f}MiB" if isinstance(peak, (int, float)) else None,
        f"eta~{_format_eta(eta_s)}",
    ]
    msg = " ".join(part for part in parts if part is not None)
    print(msg, file=sys.stderr, flush=True)


def _format_eta(seconds: float) -> str:
    if not (seconds >= 0.0):
        return "?"
    seconds_int = int(seconds + 0.5)
    minutes, sec = divmod(seconds_int, 60)
    hours, minute = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minute:02d}:{sec:02d}"
    return f"{minute:02d}:{sec:02d}"


class MAEEncoderAdapter(nn.Module):
    def __init__(self, mae: LatentResNetMAE) -> None:
        super().__init__()
        self.mae = mae

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.mae.encode_feature_taps(images)


def _build_feature_extractor(*, args: argparse.Namespace, device: torch.device, in_channels: int) -> nn.Module:
    if args.feature_encoder == "tiny":
        return TinyFeatureEncoder(
            TinyFeatureEncoderConfig(
                in_channels=in_channels,
                base_channels=args.feature_base_channels,
                stages=args.feature_stages,
            )
        ).to(device)
    mae = LatentResNetMAE(
        LatentResNetMAEConfig(
            in_channels=in_channels,
            base_channels=args.feature_base_channels,
            stages=args.feature_stages,
            encoder_arch=args.mae_encoder_arch,
            mask_ratio=0.0,
        )
    ).to(device)
    if args.mae_encoder_path is not None:
        _load_mae_encoder_weights(mae=mae, path=Path(args.mae_encoder_path), device=device)
    return MAEEncoderAdapter(mae)


def _load_mae_encoder_weights(*, mae: LatentResNetMAE, path: Path, device: torch.device) -> None:
    encoder_prefixes = ("encoder.", "paper_stem.", "paper_encoder_stages.")
    payload = torch.load(path, map_location=device)
    state_dict: dict[str, torch.Tensor]
    export_config: dict[str, object] | None = None
    if isinstance(payload, dict) and "encoder_state_dict" in payload:
        raw_encoder_state = payload["encoder_state_dict"]
        if not isinstance(raw_encoder_state, dict):
            raise ValueError(f"Invalid encoder_state_dict payload in {path}")
        state_dict = dict(raw_encoder_state)
        raw_config = payload.get("config")
        if isinstance(raw_config, dict):
            export_config = raw_config
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        model_state = payload["model_state_dict"]
        if not isinstance(model_state, dict):
            raise ValueError(f"Invalid model_state_dict payload in {path}")
        state_dict = {key: value for key, value in model_state.items() if key.startswith(encoder_prefixes)}
    elif isinstance(payload, dict):
        state_dict = {key: value for key, value in payload.items() if key.startswith(encoder_prefixes)}
    else:
        raise ValueError(f"Unsupported MAE encoder payload type in {path}: {type(payload).__name__}")
    if export_config is not None:
        _validate_mae_export_config(mae=mae, export_config=export_config, path=path)
    if not state_dict:
        raise ValueError(f"No MAE encoder weights found in {path}")
    load_result = mae.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        raise ValueError(f"Unexpected MAE encoder keys in {path}: {load_result.unexpected_keys}")


def _validate_mae_export_config(
    *,
    mae: LatentResNetMAE,
    export_config: dict[str, object],
    path: Path,
) -> None:
    expected = {
        "in_channels": int(mae.config.in_channels),
        "base_channels": int(mae.config.base_channels),
        "stages": int(mae.config.stages),
    }
    for key, expected_value in expected.items():
        if key not in export_config:
            continue
        actual_value = int(export_config[key])
        if actual_value != expected_value:
            raise ValueError(
                f"MAE export config mismatch for '{key}' in {path}: "
                f"expected {expected_value}, found {actual_value}"
            )
    if "encoder_arch" in export_config:
        actual_arch = str(export_config["encoder_arch"])
        if actual_arch != str(mae.config.encoder_arch):
            raise ValueError(
                f"MAE export config mismatch for 'encoder_arch' in {path}: "
                f"expected {mae.config.encoder_arch}, found {actual_arch}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 grouped latent smoke trainer")
    parser.add_argument("--config", type=str, default=None, help="Optional simple key:value config file")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional artifact output directory")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--keep-last-k-checkpoints", type=int, default=0)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--allow-resume-config-mismatch", action="store_true")
    parser.add_argument("--resume-model-only", action="store_true")
    parser.add_argument("--resume-reset-scheduler", action="store_true")
    parser.add_argument("--resume-reset-optimizer-lr", action="store_true")
    parser.add_argument("--compile-generator", action="store_true")
    parser.add_argument("--compile-backend", type=str, default="inductor")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-fail-action", choices=("warn", "raise", "disable"), default="warn")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    add_device_argument(parser, default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--negatives-per-group", type=int, default=4)
    parser.add_argument("--positives-per-group", type=int, default=4)
    parser.add_argument("--alpha-fixed", type=float, default=None)
    parser.add_argument("--alpha-min", type=float, default=1.0)
    parser.add_argument("--alpha-max", type=float, default=4.0)
    parser.add_argument(
        "--alpha-dist",
        choices=("uniform", "powerlaw", "mixture_point_powerlaw", "table8_l2_latent"),
        default="uniform",
        help="How to sample training-time CFG alpha when --alpha-fixed is not set.",
    )
    parser.add_argument("--alpha-power", type=float, default=3.0, help="Power-law exponent for p(alpha) ∝ alpha^-k.")
    parser.add_argument("--alpha-point", type=float, default=1.0, help="Point-mass alpha for mixture sampling.")
    parser.add_argument("--alpha-point-prob", type=float, default=0.5, help="Probability of choosing alpha-point.")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--ffn-inner-dim", type=int, default=None)
    parser.add_argument("--register-tokens", type=int, default=16)
    parser.add_argument("--style-vocab-size", type=int, default=64)
    parser.add_argument("--style-token-count", type=int, default=32)
    parser.add_argument("--norm-type", type=str, default="layernorm")
    parser.add_argument("--use-qk-norm", action="store_true")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--drift-temperatures", nargs="*", type=float, default=[])
    parser.add_argument("--drift-temperature-reduction", choices=("mean", "sum"), default="sum")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--scheduler", choices=("none", "constant", "cosine", "warmup_cosine"), default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--use-feature-loss", action="store_true")
    parser.add_argument("--feature-encoder", choices=("tiny", "mae"), default="tiny")
    parser.add_argument("--mae-encoder-path", type=str, default=None)
    parser.add_argument(
        "--mae-encoder-arch",
        choices=("resnet_unet", "legacy_conv", "paper_resnet34_unet"),
        default="resnet_unet",
    )
    parser.add_argument("--feature-base-channels", type=int, default=16)
    parser.add_argument("--feature-stages", type=int, default=3)
    parser.add_argument("--latent-feature-decode-mode", choices=("none", "identity", "conv"), default="none")
    parser.add_argument("--latent-decoder-out-channels", type=int, default=3)
    parser.add_argument("--latent-decoder-image-size", type=int, default=32)
    parser.add_argument("--latent-decoder-hidden-channels", type=int, default=64)
    parser.add_argument("--feature-temperatures", nargs="+", type=float, default=[0.02, 0.05, 0.2])
    parser.add_argument(
        "--feature-temperature-aggregation",
        choices=("per_temperature_mse", "sum_drifts_then_mse"),
        default="sum_drifts_then_mse",
    )
    parser.add_argument(
        "--feature-loss-term-reduction",
        choices=("sum", "mean"),
        default="sum",
    )
    parser.add_argument("--feature-selected-stages", nargs="*", type=int, default=[])
    parser.add_argument("--disable-per-location", action="store_true")
    parser.add_argument("--disable-global-stats", action="store_true")
    parser.add_argument("--disable-patch2-stats", action="store_true")
    parser.add_argument("--include-patch4-stats", action="store_true")
    parser.add_argument("--include-input-x2-mean", action="store_true")
    parser.add_argument("--disable-shared-location-normalization", action="store_true")
    parser.add_argument("--disable-feature-temperature-sqrt-scaling", action="store_true")
    parser.add_argument("--feature-include-raw-drift-loss", action="store_true")
    parser.add_argument("--feature-raw-drift-loss-weight", type=float, default=1.0)
    parser.add_argument("--feature-compile-drift-kernel", action="store_true")
    parser.add_argument("--feature-compile-backend", type=str, default="inductor")
    parser.add_argument("--feature-compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--feature-compile-dynamic", action="store_true")
    parser.add_argument("--feature-compile-fullgraph", action="store_true")
    parser.add_argument("--feature-compile-fail-action", choices=("warn", "raise"), default="warn")
    parser.add_argument("--use-queue", action="store_true")
    parser.add_argument("--unconditional-per-group", type=int, default=2)
    parser.add_argument("--queue-prime-samples", type=int, default=500)
    parser.add_argument("--queue-push-batch", type=int, default=64)
    parser.add_argument("--queue-per-class-capacity", type=int, default=128)
    parser.add_argument("--queue-global-capacity", type=int, default=1000)
    parser.add_argument("--queue-store-device", type=str, default="cpu")
    parser.add_argument("--queue-warmup-mode", type=str, choices=("random", "class_balanced"), default="random")
    parser.add_argument("--queue-warmup-min-per-class", type=int, default=1)
    parser.add_argument("--queue-strict-without-replacement", action="store_true")
    parser.add_argument("--queue-report-level", choices=("basic", "full"), default="basic")
    parser.add_argument("--queue-refill-policy", type=str, choices=("per_step", "every_n_steps"), default="per_step")
    parser.add_argument("--queue-refill-every", type=int, default=1)
    parser.add_argument("--real-batch-source", type=str, default="synthetic_dataset")
    parser.add_argument("--real-dataset-size", type=int, default=4096)
    parser.add_argument("--real-loader-batch-size", type=int, default=128)
    parser.add_argument("--real-num-workers", type=int, default=0)
    parser.add_argument("--disable-real-shuffle", action="store_true")
    parser.add_argument("--real-pin-memory", action="store_true")
    parser.add_argument("--real-persistent-workers", action="store_true")
    parser.add_argument("--real-prefetch-factor", type=int, default=0)
    parser.add_argument("--real-sanity-sample-batches", type=int, default=0)
    parser.add_argument("--real-imagefolder-root", type=str, default=None)
    parser.add_argument("--real-webdataset-urls", type=str, default=None)
    parser.add_argument("--real-tensor-file-path", type=str, default=None)
    parser.add_argument("--real-tensor-shards-manifest-path", type=str, default=None)
    parser.add_argument("--real-transform-resize", type=int, default=None)
    parser.add_argument("--disable-real-center-crop", action="store_true")
    parser.add_argument("--real-horizontal-flip", action="store_true")
    parser.add_argument("--real-transform-normalize", action="store_true")
    parser.add_argument("--overfit-fixed-batch", action="store_true")
    return parser.parse_args()


def _sample_alpha(args: argparse.Namespace, *, device: torch.device) -> torch.Tensor:
    return sample_alpha(
        groups=int(args.groups),
        device=device,
        alpha_fixed=args.alpha_fixed,
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        alpha_dist=str(args.alpha_dist),
        alpha_power=float(args.alpha_power),
        alpha_point=float(args.alpha_point),
        alpha_point_prob=float(args.alpha_point_prob),
    )


def _save_checkpoints(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    queue: ClassConditionalSampleQueue | None,
    step: int,
    extra: dict[str, object],
    scheduler: object,
    scaler: object,
    ema_state_dict: dict[str, object] | None,
    force: bool = False,
) -> None:
    if not force and args.checkpoint_path is None and args.checkpoint_dir is None:
        return
    if args.checkpoint_path is not None:
        save_training_checkpoint(
            path=Path(args.checkpoint_path),
            model=model,
            optimizer=optimizer,
            step=step,
            extra=extra,
            queue_state=None if queue is None else queue.state_dict(),
            scheduler=scheduler,
            scaler=scaler,
            ema_state_dict=ema_state_dict,
        )
    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        step_path = checkpoint_dir / f"checkpoint_step_{step:08d}.pt"
        save_training_checkpoint(
            path=step_path,
            model=model,
            optimizer=optimizer,
            step=step,
            extra=extra,
            queue_state=None if queue is None else queue.state_dict(),
            scheduler=scheduler,
            scaler=scaler,
            ema_state_dict=ema_state_dict,
        )
        if args.keep_last_k_checkpoints > 0:
            _prune_checkpoint_dir(checkpoint_dir, keep_last_k=int(args.keep_last_k_checkpoints))


def _prune_checkpoint_dir(checkpoint_dir: Path, *, keep_last_k: int) -> None:
    pattern = re.compile(r"^checkpoint_step_(\d+)\.pt$")
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        candidates.append((int(match.group(1)), path))
    candidates.sort(key=lambda pair: pair[0])
    if keep_last_k <= 0 or len(candidates) <= keep_last_k:
        return
    for _, path in candidates[: -keep_last_k]:
        path.unlink()


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    config_path = Path(args.config)
    entries = _load_simple_kv_config(config_path)
    for key, raw_value in entries.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            raise ValueError(f"Unknown config key '{key}' in {config_path}")
        if attr == "ffn_inner_dim":
            setattr(args, attr, int(raw_value))
            continue
        if attr == "feature_temperatures":
            setattr(args, attr, _parse_list(raw_value, caster=float))
            continue
        if attr == "drift_temperatures":
            setattr(args, attr, _parse_list(raw_value, caster=float))
            continue
        if attr == "feature_selected_stages":
            setattr(args, attr, _parse_list(raw_value, caster=int))
            continue
        current = getattr(args, attr)
        setattr(args, attr, _coerce_like(raw_value, current))
    return args


def _load_simple_kv_config(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line: {raw_line}")
        key, value = line.split(":", 1)
        entries[key.strip()] = value.strip()
    return entries


def _coerce_like(raw_value: str, template: object) -> object:
    if template is None:
        return raw_value
    if isinstance(template, bool):
        return _parse_bool(raw_value)
    if isinstance(template, int):
        return int(raw_value)
    if isinstance(template, float):
        return float(raw_value)
    if isinstance(template, str):
        return raw_value
    raise ValueError(f"Unsupported config value type: {type(template).__name__}")


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_list(value: str, *, caster: type[float] | type[int]) -> list[float] | list[int]:
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
    if not stripped:
        return []
    if "," in stripped:
        parts = [part.strip() for part in stripped.split(",")]
    else:
        parts = [part.strip() for part in stripped.split()]
    return [caster(part) for part in parts if part]


def _maybe_compile_generator_forward(
    *,
    model: nn.Module,
    args: argparse.Namespace,
) -> dict[str, object]:
    try:
        model_param = next(model.parameters())
        model_device = model_param.device
    except StopIteration:
        model_device = torch.device("cpu")
    compiled_forward, compile_result = maybe_compile_callable(
        model.forward,
        enabled=bool(args.compile_generator),
        backend=str(args.compile_backend),
        mode=str(args.compile_mode),
        dynamic=bool(args.compile_dynamic),
        fullgraph=bool(args.compile_fullgraph),
        fail_action=str(args.compile_fail_action),
        device=model_device,
        context="train_latent.compile_generator",
    )
    if compile_result.enabled:
        model.forward = compiled_forward
    return compile_result.to_dict()


def _attach_loss_scale_metrics(stats: dict[str, float]) -> None:
    loss = stats.get("loss")
    term_count = stats.get("loss_term_count")
    if isinstance(loss, (int, float)) and isinstance(term_count, (int, float)) and term_count > 0.0:
        stats["loss_per_term"] = float(loss) / float(term_count)
    feature_loss = stats.get("feature_loss")
    if isinstance(feature_loss, (int, float)) and isinstance(term_count, (int, float)) and term_count > 0.0:
        stats["feature_loss_per_term"] = float(feature_loss) / float(term_count)


def _mean_numeric_stats(entries: list[dict[str, float]]) -> dict[str, float]:
    if not entries:
        raise ValueError("entries must be non-empty")
    buckets: dict[str, list[float]] = {}
    for entry in entries:
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                buckets.setdefault(key, []).append(float(value))
    output: dict[str, float] = {}
    for key, values in buckets.items():
        output[key] = float(sum(values) / len(values))
    return output


def _resume_fingerprint_payload(args: argparse.Namespace) -> dict[str, object]:
    excluded = {
        "resume_from",
        "allow_resume_config_mismatch",
        "output_dir",
        "checkpoint_path",
        "save_every",
        "log_every",
        "steps",
    }
    payload = {key: value for key, value in vars(args).items() if key not in excluded}
    return {"args": payload}


def _build_lr_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if scheduler_name == "none":
        return None
    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    if scheduler_name == "warmup_cosine":
        warmup = max(0, warmup_steps)
        cosine_steps = max(1, total_steps - warmup)

        def lr_lambda(step: int) -> float:
            if step < warmup and warmup > 0:
                return float(step + 1) / float(warmup)
            progress = min(max(step - warmup, 0), cosine_steps)
            cosine = 0.5 * (1.0 + math.cos(progress / cosine_steps * math.pi))
            return float(cosine)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _sample_synthetic_positives(
    *,
    groups: int,
    positives_per_group: int,
    channels: int,
    image_size: int,
    class_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    base = torch.randn(groups, positives_per_group, channels, image_size, image_size, device=device)
    class_scalar = class_labels.float() / max(1.0, float(num_classes - 1))
    class_offset = class_scalar.view(groups, 1, 1, 1, 1) * 0.15
    return base + class_offset


def _prime_queue(
    *,
    queue: ClassConditionalSampleQueue,
    provider: RealBatchProvider,
    num_classes: int,
    sample_count: int,
    warmup_mode: str,
    warmup_min_per_class: int,
    device: torch.device,
) -> None:
    images, labels = _sample_real_batch(provider=provider, count=sample_count, device=device)
    if warmup_mode != "class_balanced":
        queue.push(images, labels)
        return
    if warmup_min_per_class <= 0:
        raise ValueError("queue_warmup_min_per_class must be > 0")
    target_classes = max(1, min(num_classes, sample_count // warmup_min_per_class))
    selected_images: list[torch.Tensor] = []
    selected_labels: list[torch.Tensor] = []
    per_class_counts = [0 for _ in range(target_classes)]
    missing = {index for index in range(target_classes)}
    attempts = 0
    max_attempts = max(8, target_classes * 4)
    while missing and attempts < max_attempts:
        if attempts > 0:
            images, labels = _sample_real_batch(provider=provider, count=sample_count, device=device)
        for sample_index in range(labels.shape[0]):
            label = int(labels[sample_index].item())
            if label < 0 or label >= target_classes:
                continue
            if per_class_counts[label] >= warmup_min_per_class:
                continue
            selected_images.append(images[sample_index : sample_index + 1])
            selected_labels.append(labels[sample_index : sample_index + 1])
            per_class_counts[label] += 1
            if per_class_counts[label] >= warmup_min_per_class and label in missing:
                missing.remove(label)
            if not missing:
                break
        attempts += 1
    if missing:
        missing_labels = sorted(missing)
        raise RuntimeError(
            "queue_warmup_mode=class_balanced could not satisfy per-class coverage with true labels; "
            f"missing labels: {missing_labels[:16]}{'...' if len(missing_labels) > 16 else ''}"
        )
    required = target_classes * warmup_min_per_class
    selected_batch = torch.cat(selected_images, dim=0)
    selected_batch_labels = torch.cat(selected_labels, dim=0)
    if selected_batch.shape[0] != required:
        raise RuntimeError("queue warmup selected sample count mismatch")
    remaining_capacity = sample_count - required
    if remaining_capacity > 0:
        extra_images, extra_labels = _sample_real_batch(provider=provider, count=remaining_capacity, device=device)
        selected_batch = torch.cat([selected_batch, extra_images], dim=0)
        selected_batch_labels = torch.cat([selected_batch_labels, extra_labels], dim=0)
    queue.push(selected_batch, selected_batch_labels)


def _queue_covered_class_count(queue: ClassConditionalSampleQueue) -> int:
    return sum(1 for count in queue.class_counts() if count > 0)


def _ensure_queue_has_labels(
    *,
    queue: ClassConditionalSampleQueue,
    class_labels: torch.Tensor,
    provider: RealBatchProvider,
    required_count: int = 1,
    device: torch.device,
) -> int:
    if required_count <= 0:
        raise ValueError("required_count must be > 0")
    underfilled = sorted(
        {
            int(label.item())
            for label in class_labels
            if queue.class_count(int(label.item())) < int(required_count)
        }
    )
    if not underfilled:
        return 0
    remaining = set(underfilled)
    attempts = 0
    max_attempts = max(16, len(underfilled) * 16)
    while remaining and attempts < max_attempts:
        refill_count = max(64, len(remaining) * max(16, int(required_count) * 8))
        images, labels = _sample_real_batch(provider=provider, count=refill_count, device=device)
        required = torch.tensor(sorted(remaining), device=device, dtype=torch.long)
        keep_mask = torch.isin(labels, required)
        if keep_mask.any():
            queue.push(images[keep_mask], labels[keep_mask])
            remaining = {label for label in remaining if queue.class_count(label) < int(required_count)}
        attempts += 1
    if remaining:
        unresolved = sorted(remaining)
        raise RuntimeError(
            "Queue underflow backfill could not satisfy required per-class count from real provider; "
            f"required_count={int(required_count)}, unresolved labels: "
            f"{unresolved[:16]}{'...' if len(unresolved) > 16 else ''}"
        )
    return len(underfilled)


def _sample_real_batch(
    *,
    provider: RealBatchProvider,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("count must be > 0")
    images_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []
    total = 0
    while total < count:
        images, labels = provider.next_batch(device=device)
        images_chunks.append(images)
        labels_chunks.append(labels)
        total += images.shape[0]
    images_cat = torch.cat(images_chunks, dim=0)[:count]
    labels_cat = torch.cat(labels_chunks, dim=0)[:count]
    return images_cat, labels_cat


def _build_queue_warmup_report(
    *,
    queue: ClassConditionalSampleQueue,
    warmup_mode: str,
    samples_pushed: int,
    report_level: str,
) -> dict[str, object]:
    counts = queue.class_counts()
    non_zero = [count for count in counts if count > 0]
    report: dict[str, object] = {
        "warmup_mode": warmup_mode,
        "samples_pushed": float(samples_pushed),
        "global_count": float(queue.global_count()),
        "covered_classes": float(len(non_zero)),
        "min_class_count_nonzero": float(min(non_zero) if non_zero else 0),
        "max_class_count": float(max(counts) if counts else 0),
        "mean_class_count": float(sum(counts) / len(counts) if counts else 0),
    }
    if report_level == "full":
        sorted_non_zero = sorted(non_zero)
        def _quantile(values: list[int], q: float) -> float:
            if not values:
                return 0.0
            index = int(round(q * (len(values) - 1)))
            return float(values[max(0, min(len(values) - 1, index))])
        report["counts"] = counts
        report["count_quantiles_nonzero"] = {
            "p10": _quantile(sorted_non_zero, 0.10),
            "p50": _quantile(sorted_non_zero, 0.50),
            "p90": _quantile(sorted_non_zero, 0.90),
        }
    return report


def _build_real_provider_sanity_report(
    *,
    provider: RealBatchProvider,
    sample_batches: int,
    num_classes: int,
    device: torch.device,
) -> dict[str, object]:
    if sample_batches <= 0:
        raise ValueError("sample_batches must be > 0")
    labels: list[torch.Tensor] = []
    images_dtype = None
    for _ in range(sample_batches):
        images, y = provider.next_batch(device=device)
        labels.append(y.detach().cpu().long())
        if images_dtype is None:
            images_dtype = str(images.dtype)
    labels_all = torch.cat(labels, dim=0)
    counts = torch.bincount(labels_all, minlength=int(num_classes))
    covered = int((counts > 0).sum().item())
    non_zero = counts[counts > 0]
    return {
        "sample_batches": int(sample_batches),
        "samples": int(labels_all.numel()),
        "images_dtype": images_dtype,
        "label_min": int(labels_all.min().item()) if labels_all.numel() else None,
        "label_max": int(labels_all.max().item()) if labels_all.numel() else None,
        "covered_classes": int(covered),
        "min_class_count_nonzero": int(non_zero.min().item()) if non_zero.numel() else 0,
        "max_class_count": int(counts.max().item()) if counts.numel() else 0,
        "mean_class_count": float(counts.float().mean().item()) if counts.numel() else 0.0,
    }


def _assert_single_process() -> None:
    import os

    raw_world_size = os.environ.get("WORLD_SIZE", "").strip()
    if not raw_world_size:
        return
    try:
        world_size = int(raw_world_size)
    except ValueError:
        world_size = 1
    if world_size > 1:
        raise RuntimeError(
            "Distributed execution is not supported for this training entrypoint (WORLD_SIZE>1). "
            "Run single-process (WORLD_SIZE=1) or implement a validated DDP strategy first."
        )


if __name__ == "__main__":
    main()
