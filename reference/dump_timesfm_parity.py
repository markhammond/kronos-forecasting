"""Dump reference tensors so the .NET port can be bisected stage by stage.

A single end-to-end comparison cannot tell a wrong RoPE ordering from a wrong norm
placement — both just shift the logits. Capturing every stage boundary localises the
first divergence instead.

Run: .venv/bin/python reference/dump_parity.py
"""

import json
import sys
import pathlib

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from timesfm3 import model as tfm_model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "timesfm-3.0-pytorch"
OUT = ROOT / "reference" / "parity.safetensors"

# Small enough to compare by hand, large enough to exercise variate attention,
# causal masking and the running-stats recurrence across several patches.
B, V, N, P = 1, 3, 6, 32

torch.manual_seed(20260901)


def main() -> None:
    cfg = json.loads((CKPT / "config.json").read_text())
    net = tfm_model.TimesFM3Torch(
        input_patch_len=cfg["input_patch_len"],
        output_patch_len=cfg["output_patch_len"],
        quantiles=cfg["quantiles"],
        residual_block_config=cfg["residual_block_config"],
        transformer_config=cfg["transformer_config"],
        use_variate_attention=cfg["use_variate_attention"],
        value_clip=cfg["value_clip"],
        use_stitching=cfg["use_stitching"],
        use_linear_detrending=cfg["use_linear_detrending"],
        linear_detrending_threshold=cfg["linear_detrending_threshold"],
        use_iterative_cpm_revin=cfg["use_iterative_cpm_revin"],
        use_frozen_running_stats=cfg["use_frozen_running_stats"],
        input_transform=cfg["input_transform"],
    )
    state = load_file(str(CKPT / "model.safetensors"))
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"  loaded: {len(state)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
    if missing:
        print(f"    missing[:5]: {missing[:5]}")
    if unexpected:
        print(f"    unexpected[:5]: {unexpected[:5]}")
    net.eval()

    # A deterministic, non-degenerate input: distinct per-variate scale and drift so a
    # transposed variate axis or a dropped running-stat cannot pass unnoticed.
    t = torch.arange(N * P, dtype=torch.float32).reshape(1, 1, N, P)
    scale = torch.tensor([1.0, 7.5, 0.25]).reshape(1, V, 1, 1)
    drift = torch.tensor([0.0, -3.0, 2.0]).reshape(1, V, 1, 1)
    values = (torch.sin(t / 11.0) * scale + drift + t * 0.01).expand(B, V, N, P).contiguous()
    values = values + torch.randn(B, V, N, P) * 0.05

    masks = torch.zeros(B, V, N, P, dtype=torch.bool)
    masks[:, :, 0, :16] = True          # leading partial mask exercises left-padding
    patch_is_target = torch.zeros(B, V, N, dtype=torch.bool)
    patch_is_target[:, 0, :] = True     # variate 0 is the target

    out = {"in_values": values, "in_masks": masks.float(),
           "in_patch_is_target": patch_is_target.float()}

    # Preprocessing boundaries.
    run_n, run_mu, run_sigma = tfm_model.util.get_running_stats(values, masks)
    out |= {"running_n": run_n, "running_mu": run_mu, "running_sigma": run_sigma}

    rb_in, rb_out, patch_mask, (rv_mu, rv_sd), _ = net._preprocess(
        values, masks, patch_is_target)
    out |= {"resblock_input": rb_in, "resblock_output": rb_out,
            "patch_mask": patch_mask.float()}

    # Per-layer outputs, so a divergence is attributable to one layer.
    eff_mask = torch.cumprod(patch_mask.int(), dim=2).bool()
    h = rb_out
    with torch.no_grad():
        for i, layer in enumerate(net.transformer_stack.layers):
            h, _, _ = layer(h, eff_mask)
            if i in (0, 1, 19):
                out[f"layer{i}_output"] = h.clone()
        out["transformer_output"] = h
        out["raw_logits"] = net.output_head(h)

    save_file({k: v.contiguous() for k, v in out.items()}, str(OUT))
    print(f"  wrote {OUT} ({len(out)} tensors)")
    for k in ("resblock_input", "resblock_output", "raw_logits"):
        print(f"    {k:20s} {tuple(out[k].shape)}")


if __name__ == "__main__":
    main()
