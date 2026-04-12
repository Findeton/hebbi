"""
Energy consolidation training for Hebbi — teach the model to use recurrent
energy dynamics (energy_steps > 1).

Loads an SFT checkpoint (trained with energy_steps=1) and fine-tunes with
multi-pass energy iteration. Each training step:

  1. Embed tokens
  2. Run the full block stack E times (default 3), feeding each pass's output
     back as the next pass's input
  3. Accumulate per-block FF gradients across all E passes (weighted by
     --energy-weight-mode), but do NOT step optimizers until all passes finish
  4. Step each block optimizer once — weights stay fixed across passes within
     one training iteration, matching inference behavior

This teaches the model's representational space to be self-consistent: the
output of pass E should be a useful input for pass E+1.

    python -m scripts.train_energy --checkpoint=checkpoints/hebbi_sft_final.pt
    python -m scripts.train_energy --checkpoint=checkpoints/hebbi_sft_final.pt --energy-steps=3 --energy-weight-mode=increasing
"""

import os
import time
import json
import argparse
from dataclasses import asdict

import torch

from hebbi.model import DET, DETConfig
from hebbi.local_learning import (
    generate_negatives,
    det_train_step_with_energy,
    compute_energy_weights,
    LayerOptimizers,
)
from hebbi.data import get_tokenizer, get_data_loader
from hebbi.common import (
    compute_init, autodetect_device_type, print0, DummyWandb, COMPUTE_DTYPE,
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Energy consolidation training for Hebbi")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="SFT (or pretrain) checkpoint path")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--dataset", type=str, default="smoltalk",
                    help="dataset for energy training")
# Energy
parser.add_argument("--energy-steps", type=int, default=3,
                    help="number of recurrent energy iterations per training step")
parser.add_argument("--energy-weight-mode", type=str, default="increasing",
                    choices=["increasing", "uniform", "decreasing", "final_only", "custom"],
                    help="how to weight per-stage gradients")
parser.add_argument("--energy-weights-custom", type=str, default="",
                    help="comma-separated custom weights (required when mode=custom)")
parser.add_argument("--energy-threshold", type=float, default=0.01,
                    help="convergence threshold for dynamic stopping at inference (0=always run all steps)")
# Training
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--num-iterations", type=int, default=2000)
parser.add_argument("--block-lr", type=float, default=1e-4,
                    help="block LR (lower than pretrain — fine-tuning)")
parser.add_argument("--head-lr", type=float, default=1e-4)
parser.add_argument("--embed-lr", type=float, default=3e-4)
parser.add_argument("--corruption-rate", type=float, default=0.15)
parser.add_argument("--warmup-steps", type=int, default=100)
parser.add_argument("--warmdown-ratio", type=float, default=0.3)
parser.add_argument("--save-every", type=int, default=500)
parser.add_argument("--sample-every", type=int, default=500)
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = compute_init(device_type)

# Wandb
try:
    import wandb as wandb_lib
    use_wandb = args.run != "dummy"
except ImportError:
    use_wandb = False
wandb_run = DummyWandb()
if use_wandb:
    wandb_run = wandb_lib.init(project="hebbi-energy", name=args.run, config=vars(args))

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
print0(f"Loading checkpoint: {args.checkpoint}")
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
config = DETConfig(**checkpoint["config"])

# Override energy config to match training
config.energy_steps = args.energy_steps
config.energy_threshold = args.energy_threshold

model = DET(config).to(device)
model.load_state_dict(checkpoint["model"], strict=False)
n_params = sum(p.numel() for p in model.parameters())
print0(f"Config: {json.dumps(asdict(config), indent=2)}")
print0(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

if args.compile and hasattr(torch, "compile"):
    print0("Compiling model...")
    model = torch.compile(model)

# ---------------------------------------------------------------------------
# Energy weights
# ---------------------------------------------------------------------------
if args.energy_weight_mode == "custom":
    custom = [float(x) for x in args.energy_weights_custom.split(",")]
    energy_weights = compute_energy_weights(
        args.energy_steps, mode="custom", custom_weights=custom
    )
else:
    energy_weights = compute_energy_weights(
        args.energy_steps, mode=args.energy_weight_mode
    )

print0(f"Energy steps: {args.energy_steps}")
print0(f"Weight mode: {args.energy_weight_mode}")
print0(f"Weights (normalized): [{', '.join(f'{w:.3f}' for w in energy_weights)}]")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
tokenizer = get_tokenizer()
print0(f"Dataset: {args.dataset} | Vocab: {tokenizer.get_vocab_size()}")
train_loader, _ = get_data_loader(
    args.dataset, tokenizer, args.batch_size, config.sequence_len, device, "train"
)

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
layer_opts = LayerOptimizers(model, args.block_lr, args.head_lr, args.embed_lr)

neg_rng = torch.Generator(device="cpu")
neg_rng.manual_seed(1337)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print0(f"\nEnergy consolidation for {args.num_iterations} steps...")
print0(f"Each step: {args.energy_steps} passes through block stack, "
       f"one optimizer step at the end")
print0()

smooth_lm = 0.0
smooth_ff = 0.0
t_start = time.time()
tokens_processed = 0

for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations

    # --- Sample (compare single-pass vs multi-pass generation) ---
    if args.sample_every > 0 and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompt_text = "Once upon a time"
        prompt_ids = tokenizer.encode(prompt_text)

        gen = list(model.generate(prompt_ids, max_tokens=200, temperature=0.8, top_k=40))
        text_out = prompt_text + tokenizer.decode(gen)
        print0(f"--- sample @ step {step} ---")
        print0(text_out[:500])
        # Report convergence diagnostics from the last forward pass
        iters = getattr(model, "_last_energy_iters", config.energy_steps)
        delta = getattr(model, "_last_energy_delta", 0.0)
        print0(f"--- energy: {iters}/{config.energy_steps} iters, delta={delta:.6f}, threshold={config.energy_threshold}")
        model.train()

    if last_step:
        break

    # --- Training step ---
    t0 = time.time()
    lrm = layer_opts.update_lr(step, args.num_iterations,
                                args.warmup_steps, args.warmdown_ratio)

    batch = next(train_loader)
    x = batch[0]
    neg_ids = generate_negatives(x, config.vocab_size, config.corruption_rate, neg_rng)

    losses = det_train_step_with_energy(
        model, x, neg_ids, layer_opts, config, energy_weights
    )

    tokens_processed += x.numel()
    dt = time.time() - t0

    # Smoothed losses
    ema = 0.95
    lm_loss = losses["lm_loss"]
    ff_avg = sum(v for k, v in losses.items()
                 if k.startswith("ff_") and "_e" not in k) / config.n_layer
    smooth_lm = ema * smooth_lm + (1 - ema) * lm_loss
    smooth_ff = ema * smooth_ff + (1 - ema) * ff_avg
    debiased_lm = smooth_lm / (1 - ema ** (step + 1))
    debiased_ff = smooth_ff / (1 - ema ** (step + 1))

    if step % 10 == 0:
        g_pos_avg = sum(v for k, v in losses.items()
                        if k.startswith("goodness_pos_") and "_e" not in k
                        ) / config.n_layer
        g_neg_avg = sum(v for k, v in losses.items()
                        if k.startswith("goodness_neg_") and "_e" not in k
                        ) / config.n_layer
        # Per-energy-stage diagnostics: FF loss, LM loss, convergence delta
        per_stage = []
        E = args.energy_steps
        for e in range(E):
            stage_ff = sum(losses[f"ff_{i}_e{e}"] for i in range(config.n_layer)) / config.n_layer
            parts = [f"e{e}: ff={stage_ff:.3f}"]
            if f"lm_loss_e{e}" in losses:
                parts.append(f"lm={losses[f'lm_loss_e{e}']:.3f}")
            if f"energy_delta_e{e}" in losses:
                parts.append(f"d={losses[f'energy_delta_e{e}']:.4f}")
            per_stage.append(" ".join(parts))
        stage_str = " | ".join(per_stage)

        elapsed = time.time() - t_start
        toks_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        print0(
            f"step {step:05d}/{args.num_iterations} | "
            f"lm: {debiased_lm:.3f} | ff: {debiased_ff:.3f} [{stage_str}] | "
            f"g+: {g_pos_avg:.2f} g-: {g_neg_avg:.2f} | "
            f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | "
            f"tok/s: {toks_per_sec:.0f}"
        )

    if step % 50 == 0:
        log_dict = {
            "step": step,
            "energy/lm_loss": debiased_lm,
            "energy/ff_avg": debiased_ff,
            "energy/lrm": lrm,
        }
        for k, v in losses.items():
            log_dict[f"energy/{k}"] = v
        wandb_run.log(log_dict)

    # --- Checkpoint ---
    if args.save_every > 0 and step > 0 and step % args.save_every == 0:
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/hebbi_energy_{step:06d}.pt"
        torch.save({
            "config": asdict(config),
            "model": model.state_dict(),
            "step": step,
            "stage": "energy",
        }, path)
        print0(f"Saved: {path}")

# Final save
os.makedirs("checkpoints", exist_ok=True)
path = "checkpoints/hebbi_energy_final.pt"
torch.save({
    "config": asdict(config),
    "model": model.state_dict(),
    "step": args.num_iterations,
    "stage": "energy",
}, path)
elapsed = time.time() - t_start
print0(f"\nEnergy consolidation complete in {elapsed:.0f}s. Checkpoint: {path}")
print0(f"Total tokens processed: {tokens_processed:,}")
print0(f"Energy steps: {args.energy_steps} | Weights: {energy_weights}")
wandb_run.finish()
