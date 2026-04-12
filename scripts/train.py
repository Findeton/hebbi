"""
Train DET model. From the det/ project root, run as:

    python -m scripts.train

Smaller config for CPU/Mac:

    python -m scripts.train --depth=3 --n-embd=64 --seq-len=64 --batch-size=4 --num-iterations=100

Full training run:

    python -m scripts.train --depth=6 --num-iterations=5000 --run=det_v1
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
    det_train_step,
    LayerOptimizers,
)
from hebbi.data import get_shakespeare, CharDataset, char_data_loader
from hebbi.common import compute_init, autodetect_device_type, print0, DummyWandb, COMPUTE_DTYPE

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train DET model")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty=auto)")
# Model
parser.add_argument("--depth", type=int, default=6, help="number of local blocks")
parser.add_argument("--n-embd", type=int, default=256, help="embedding dimension")
parser.add_argument("--n-head", type=int, default=4, help="number of attention heads")
parser.add_argument("--seq-len", type=int, default=256, help="context length")
parser.add_argument("--hopfield-steps", type=int, default=3, help="energy attention iterations")
parser.add_argument("--hopfield-beta", type=float, default=1.0, help="inverse temperature")
parser.add_argument("--ff-threshold", type=float, default=2.0, help="Forward-Forward goodness threshold")
parser.add_argument("--energy-steps", type=int, default=1, help="recurrent thinking iterations")
parser.add_argument("--n-mem", type=int, default=0, help="GradMem prefix tokens (0=disabled)")
# Training
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument("--num-iterations", type=int, default=5000, help="training steps")
parser.add_argument("--block-lr", type=float, default=1e-3, help="learning rate for blocks")
parser.add_argument("--head-lr", type=float, default=1e-3, help="learning rate for LM head")
parser.add_argument("--embed-lr", type=float, default=1e-2, help="learning rate for embedding")
parser.add_argument("--corruption-rate", type=float, default=0.15, help="token corruption rate")
parser.add_argument("--warmup-steps", type=int, default=40, help="LR warmup steps")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="LR warmdown ratio")
# Eval
parser.add_argument("--eval-every", type=int, default=250, help="eval frequency (-1=disable)")
parser.add_argument("--sample-every", type=int, default=500, help="sample frequency (-1=disable)")
parser.add_argument("--save-every", type=int, default=-1, help="checkpoint frequency (-1=only end)")
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
    wandb_run = wandb_lib.init(project="hebbi", name=args.run, config=vars(args))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
text = get_shakespeare()
dataset = CharDataset(text)
print0(f"Vocab size: {dataset.vocab_size} | Data: {len(dataset.data):,} chars")

train_loader = char_data_loader(dataset.data, args.batch_size, args.seq_len, device, "train")
val_loader_fn = lambda: char_data_loader(dataset.data, args.batch_size, args.seq_len, device, "val")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
config = DETConfig(
    sequence_len=args.seq_len,
    vocab_size=dataset.vocab_size,
    n_layer=args.depth,
    n_embd=args.n_embd,
    n_head=args.n_head,
    hopfield_beta=args.hopfield_beta,
    hopfield_steps=args.hopfield_steps,
    ff_threshold=args.ff_threshold,
    energy_steps=args.energy_steps,
    n_mem=args.n_mem,
    corruption_rate=args.corruption_rate,
)
model = DET(config).to(device)
model.init_weights()
n_params = sum(p.numel() for p in model.parameters())
print0(f"Config: {json.dumps(asdict(config), indent=2)}")
print0(f"Parameters: {n_params:,}")

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
layer_opts = LayerOptimizers(model, args.block_lr, args.head_lr, args.embed_lr)

# RNG for negatives (MPS doesn't support torch.Generator, use CPU)
neg_rng = torch.Generator(device="cpu")
neg_rng.manual_seed(1337)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print0(f"\nTraining for {args.num_iterations} steps...")
print0(f"Forward-Forward threshold: {config.ff_threshold}")
print0(f"Hopfield steps: {config.hopfield_steps} | Energy steps: {config.energy_steps}")
print0()

smooth_lm = 0.0
smooth_ff = 0.0
t_start = time.time()

for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations

    # --- Evaluation ---
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = val_loader_fn()
        val_losses = []
        for _ in range(20):
            vx, vy = next(val_loader)
            with torch.no_grad():
                val_loss = model(vx, vy)
            val_losses.append(val_loss.item())
        val_avg = sum(val_losses) / len(val_losses)
        print0(f"step {step:05d} | val_loss: {val_avg:.4f}")
        wandb_run.log({"step": step, "val/loss": val_avg})
        model.train()

    # --- Sample ---
    if args.sample_every > 0 and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompt = "ROMEO:"
        tokens = dataset.encode(prompt)
        gen = list(model.generate(tokens, max_tokens=200, temperature=0.8, top_k=40))
        text_out = prompt + dataset.decode(gen)
        print0(f"--- sample @ step {step} ---")
        print0(text_out[:500])
        print0("---")
        model.train()

    if last_step:
        break

    # --- Training step ---
    t0 = time.time()

    # Update learning rates
    lrm = layer_opts.update_lr(step, args.num_iterations, args.warmup_steps, args.warmdown_ratio)

    # Get batch and generate negatives
    x, y = next(train_loader)
    neg_ids = generate_negatives(x, config.vocab_size, config.corruption_rate, neg_rng)

    # DET train step (per-block FF + LM head)
    losses = det_train_step(model, x, neg_ids, layer_opts, config)

    dt = time.time() - t0

    # Smoothed losses
    ema = 0.95
    lm_loss = losses["lm_loss"]
    ff_avg = sum(v for k, v in losses.items() if k.startswith("ff_")) / config.n_layer
    smooth_lm = ema * smooth_lm + (1 - ema) * lm_loss
    smooth_ff = ema * smooth_ff + (1 - ema) * ff_avg
    debiased_lm = smooth_lm / (1 - ema ** (step + 1))
    debiased_ff = smooth_ff / (1 - ema ** (step + 1))

    # Log
    if step % 10 == 0:
        # Goodness separation (positive should be > negative)
        g_pos_avg = sum(v for k, v in losses.items() if k.startswith("goodness_pos")) / config.n_layer
        g_neg_avg = sum(v for k, v in losses.items() if k.startswith("goodness_neg")) / config.n_layer
        elapsed = time.time() - t_start
        print0(
            f"step {step:05d}/{args.num_iterations} | "
            f"lm: {debiased_lm:.3f} | ff: {debiased_ff:.3f} | "
            f"g+: {g_pos_avg:.2f} g-: {g_neg_avg:.2f} | "
            f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | "
            f"elapsed: {elapsed:.0f}s"
        )

    if step % 50 == 0:
        log_dict = {"step": step, "train/lm_loss": debiased_lm, "train/ff_avg": debiased_ff, "train/lrm": lrm}
        for k, v in losses.items():
            log_dict[f"train/{k}"] = v
        wandb_run.log(log_dict)

    # --- Checkpoint ---
    if args.save_every > 0 and step > 0 and step % args.save_every == 0:
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/det_{step:06d}.pt"
        torch.save({"config": asdict(config), "model": model.state_dict()}, path)
        print0(f"Saved checkpoint: {path}")

# Final save
os.makedirs("checkpoints", exist_ok=True)
path = f"checkpoints/det_final.pt"
torch.save({"config": asdict(config), "model": model.state_dict()}, path)
print0(f"\nTraining complete. Final checkpoint: {path}")
wandb_run.finish()
