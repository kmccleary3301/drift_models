from __future__ import annotations

import argparse
import json

import torch

from drifting_models.models.dit_like import DiTLikeConfig, DiTLikeGenerator


def _count_params(model: torch.nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable}


def main() -> None:
    args = _parse_args()
    config = DiTLikeConfig(
        image_size=args.image_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
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
        alpha_hidden_dim=args.alpha_hidden_dim,
        norm_type=args.norm_type,
        use_qk_norm=args.use_qk_norm,
        use_rope=args.use_rope,
    )
    model = DiTLikeGenerator(config)
    counts = _count_params(model)

    batch = int(args.batch)
    noise = torch.randn(batch, args.in_channels, args.image_size, args.image_size)
    class_labels = torch.zeros(batch, dtype=torch.long)
    alpha = torch.ones(batch)
    style = None
    if args.style_token_count > 0:
        style = torch.zeros(batch, args.style_token_count, dtype=torch.long)
    with torch.no_grad():
        out = model(noise, class_labels, alpha, style)

    report = {
        "config": config.__dict__,
        "param_counts": counts,
        "io_shapes": {
            "noise": list(noise.shape),
            "out": list(out.shape),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report DiTLikeGenerator parameter counts and IO shapes.")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--in-channels", type=int, default=4)
    p.add_argument("--out-channels", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--ffn-inner-dim", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--register-tokens", type=int, default=16)
    p.add_argument("--style-vocab-size", type=int, default=64)
    p.add_argument("--style-token-count", type=int, default=32)
    p.add_argument("--alpha-hidden-dim", type=int, default=128)
    p.add_argument("--norm-type", type=str, default="rmsnorm")
    p.add_argument("--use-qk-norm", action="store_true")
    p.add_argument("--use-rope", action="store_true")
    p.add_argument("--batch", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    main()
