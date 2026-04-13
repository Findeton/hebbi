"""
Train Hebbi model. From the hebbi/ project root:

    # Quick smoke test (CPU, Shakespeare char-level)
    python -m scripts.train --dataset=shakespeare --depth=3 --n-embd=64 --seq-len=64 --batch-size=4 --num-iterations=100

    # TinyStories (GPU)
    python -m scripts.train --dataset=tinystories --depth=6 --num-iterations=10000

    # Full 100M model on ClimbMix (GPU)
    python -m scripts.train --dataset=climbmix --depth=12

    # Resume from checkpoint
    python -m scripts.train --dataset=climbmix --depth=12 --resume=checkpoints/hebbi_005000.pt
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
    generate_negatives_predictive,
    det_train_step,
    backprop_train_step,
    backprop_step_optimizers,
    LayerOptimizers,
    AdaptiveThreshold,
)
from hebbi.data import (
    get_shakespeare, CharDataset, char_data_loader,
    get_tokenizer, get_data_loader, DATASETS,
)
from hebbi.common import compute_init, autodetect_device_type, print0, DummyWandb, COMPUTE_DTYPE

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train Hebbi model")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty=auto)")
# Dataset
parser.add_argument("--dataset", type=str, default="shakespeare",
                    choices=list(DATASETS.keys()),
                    help="dataset to train on")
# Model
parser.add_argument("--depth", type=int, default=None, help="number of local blocks (auto-config via from_depth)")
parser.add_argument("--n-embd", type=int, default=None, help="embedding dimension (overrides from_depth)")
parser.add_argument("--n-head", type=int, default=None, help="number of attention heads (overrides from_depth)")
parser.add_argument("--seq-len", type=int, default=1024, help="context length")
parser.add_argument("--hopfield-steps", type=int, default=3, help="energy attention iterations")
parser.add_argument("--hopfield-beta", type=float, default=1.0, help="inverse temperature")
parser.add_argument("--ff-threshold", type=float, default=2.0, help="Forward-Forward goodness threshold")
parser.add_argument("--energy-steps", type=int, default=1, help="recurrent thinking iterations")
parser.add_argument("--n-mem", type=int, default=0, help="GradMem prefix tokens (0=disabled)")
# Training
parser.add_argument("--batch-size", type=int, default=32, help="micro batch size")
parser.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
parser.add_argument("--num-iterations", type=int, default=5000, help="training steps")
parser.add_argument("--block-lr", type=float, default=1e-3, help="learning rate for blocks")
parser.add_argument("--head-lr", type=float, default=1e-3, help="learning rate for LM head")
parser.add_argument("--embed-lr", type=float, default=1e-2, help="learning rate for embedding")
parser.add_argument("--corruption-rate", type=float, default=0.15, help="token corruption rate")
parser.add_argument("--adaptive-threshold", action="store_true",
                    help="enable adaptive FF threshold that tracks goodness")
parser.add_argument("--threshold-margin", type=float, default=0.5,
                    help="adaptive threshold = margin * EMA(goodness_pos)")
parser.add_argument("--predictive-negatives", action="store_true",
                    help="use model predictions as negatives (predictive coding)")
parser.add_argument("--warmup-steps", type=int, default=40, help="LR warmup steps")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="LR warmdown ratio")
# Eval
parser.add_argument("--eval-every", type=int, default=250, help="eval frequency (-1=disable)")
parser.add_argument("--sample-every", type=int, default=500, help="sample frequency (-1=disable)")
parser.add_argument("--save-every", type=int, default=1000, help="checkpoint frequency (-1=only end)")
# Resume
parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume from")
# Compile
parser.add_argument("--compile", action="store_true", help="use torch.compile (requires PyTorch 2.0+)")
# Baseline
parser.add_argument("--backprop", action="store_true",
                    help="use standard backprop instead of FF (baseline comparison)")
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
is_char = args.dataset == "shakespeare"

if is_char:
    text = get_shakespeare()
    char_dataset = CharDataset(text)
    vocab_size = char_dataset.vocab_size
    print0(f"Dataset: Shakespeare (char-level) | Vocab: {vocab_size} | Data: {len(char_dataset.data):,} chars")
    train_loader = char_data_loader(char_dataset.data, args.batch_size, args.seq_len, device, "train")
    val_loader_fn = lambda: char_data_loader(char_dataset.data, args.batch_size, args.seq_len, device, "val")
else:
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size_padded()
    print0(f"Dataset: {args.dataset} | Vocab: {vocab_size} (padded) | Raw: {tokenizer.get_vocab_size()}")
    train_loader, _ = get_data_loader(args.dataset, tokenizer, args.batch_size, args.seq_len, device, "train")
    val_loader_fn = None  # streaming datasets — eval via train loss smoothing

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if args.resume:
    # Resume from checkpoint
    print0(f"Resuming from {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    config = DETConfig(**checkpoint["config"])
    model = DET(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    start_step = checkpoint.get("step", 0)
    # CLI flags override checkpoint state for these features
    # (so you can enable them mid-training by adding the flag on resume)
    if args.adaptive_threshold:
        config.ff_threshold = args.ff_threshold
    if args.corruption_rate != 0.15:  # explicit override
        config.corruption_rate = args.corruption_rate
    print0(f"Resumed at step {start_step}")
else:
    # Build config
    if args.depth is not None:
        config = DETConfig.from_depth(
            args.depth,
            sequence_len=args.seq_len,
            vocab_size=vocab_size,
            hopfield_beta=args.hopfield_beta,
            hopfield_steps=args.hopfield_steps,
            ff_threshold=args.ff_threshold,
            energy_steps=args.energy_steps,
            n_mem=args.n_mem,
            corruption_rate=args.corruption_rate,
        )
        # Allow explicit overrides
        if args.n_embd is not None:
            config.n_embd = args.n_embd
        if args.n_head is not None:
            config.n_head = args.n_head
    else:
        config = DETConfig(
            sequence_len=args.seq_len,
            vocab_size=vocab_size,
            hopfield_beta=args.hopfield_beta,
            hopfield_steps=args.hopfield_steps,
            ff_threshold=args.ff_threshold,
            energy_steps=args.energy_steps,
            n_mem=args.n_mem,
            corruption_rate=args.corruption_rate,
        )
    model = DET(config).to(device)
    model.init_weights()
    start_step = 0

n_params = sum(p.numel() for p in model.parameters())
print0(f"Config: {json.dumps(asdict(config), indent=2)}")
print0(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

# Optional torch.compile
if args.compile and hasattr(torch, "compile"):
    print0("Compiling model with torch.compile...")
    model = torch.compile(model)

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
layer_opts = LayerOptimizers(model, args.block_lr, args.head_lr, args.embed_lr)

# Adaptive threshold (optional)
adapt_thresh = None
if args.adaptive_threshold:
    adapt_thresh = AdaptiveThreshold(
        base_threshold=config.ff_threshold,
        margin_ratio=args.threshold_margin,
    )
    # Restore state from checkpoint if available
    if args.resume and "adaptive_threshold" in checkpoint:
        adapt_thresh.load_state_dict(checkpoint["adaptive_threshold"])
        # CLI flags override saved hyperparameters (so you can tune mid-training)
        adapt_thresh.margin_ratio = args.threshold_margin
        print0(f"Restored adaptive threshold: {adapt_thresh.threshold:.2f} "
               f"(EMA={adapt_thresh.goodness_ema:.2f}, n={adapt_thresh.n_updates})")
    print0(f"Adaptive threshold enabled: base={config.ff_threshold}, margin={args.threshold_margin}")

if args.predictive_negatives:
    print0("Predictive coding negatives enabled")

# RNG for negatives (MPS doesn't support torch.Generator, use CPU)
neg_rng = torch.Generator(device="cpu")
neg_rng.manual_seed(1337)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print0(f"\nTraining for {args.num_iterations} steps (grad_accum={args.grad_accum}, effective_batch={args.batch_size * args.grad_accum})...")
if args.backprop:
    print0("Mode: BACKPROP (baseline)")
else:
    print0(f"Mode: Forward-Forward | threshold: {config.ff_threshold}")
print0(f"Hopfield steps: {config.hopfield_steps} | Energy steps: {config.energy_steps}")
print0()

smooth_lm = 0.0
smooth_ff = 0.0
t_start = time.time()
tokens_processed = 0

for step in range(start_step, args.num_iterations + 1):
    last_step = step == args.num_iterations

    # --- Evaluation ---
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        if val_loader_fn is not None:
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
        if is_char:
            prompt = "ROMEO:"
            tokens = char_dataset.encode(prompt)
            gen = list(model.generate(tokens, max_tokens=200, temperature=0.8, top_k=40))
            text_out = prompt + char_dataset.decode(gen)
        else:
            prompt = "Once upon a time"
            tokens = tokenizer.encode(prompt)
            gen = list(model.generate(tokens, max_tokens=200, temperature=0.8, top_k=40))
            text_out = prompt + tokenizer.decode(gen)
        print0(f"--- sample @ step {step} ---")
        print0(text_out[:500])
        print0("---")
        model.train()

    if last_step:
        break

    # --- Training step (with gradient accumulation) ---
    t0 = time.time()

    # Update learning rates
    lrm = layer_opts.update_lr(step, args.num_iterations, args.warmup_steps, args.warmdown_ratio)

    # Gradient accumulation
    accum_losses = {}

    if args.backprop:
        # Backprop: accumulate gradients across micro-steps, step once
        layer_opts.zero_all()
        for micro_step in range(args.grad_accum):
            batch = next(train_loader)
            x = batch[0]
            losses = backprop_train_step(model, x, layer_opts, config,
                                         grad_accum=args.grad_accum)
            tokens_processed += x.numel()
            for k, v in losses.items():
                accum_losses[k] = accum_losses.get(k, 0.0) + v / args.grad_accum
        backprop_step_optimizers(model, layer_opts)
    else:
        # FF: each micro-step does its own per-block optimizer steps
        for micro_step in range(args.grad_accum):
            batch = next(train_loader)
            x = batch[0]
            if args.predictive_negatives and not is_char:
                neg_ids = generate_negatives_predictive(model, x, config.corruption_rate, neg_rng)
            else:
                neg_ids = generate_negatives(x, config.vocab_size, config.corruption_rate, neg_rng)
            losses = det_train_step(model, x, neg_ids, layer_opts, config,
                                    adaptive_threshold=adapt_thresh)
            tokens_processed += x.numel()
            for k, v in losses.items():
                accum_losses[k] = accum_losses.get(k, 0.0) + v / args.grad_accum

    dt = time.time() - t0

    # Smoothed losses
    ema = 0.95
    lm_loss = accum_losses["lm_loss"]
    ff_avg = sum(v for k, v in accum_losses.items() if k.startswith("ff_")) / max(config.n_layer, 1)
    smooth_lm = ema * smooth_lm + (1 - ema) * lm_loss
    smooth_ff = ema * smooth_ff + (1 - ema) * ff_avg
    age = step - start_step + 1
    debiased_lm = smooth_lm / (1 - ema ** age)
    debiased_ff = smooth_ff / (1 - ema ** age)

    # Log
    if step % 10 == 0:
        elapsed = time.time() - t_start
        toks_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        # Get actual LR from first block optimizer
        actual_lr = layer_opts.block_optimizers[0].param_groups[0]["lr"]
        if args.backprop:
            print0(
                f"step {step:05d}/{args.num_iterations} | "
                f"lm: {debiased_lm:.3f} | "
                f"lr: {actual_lr:.2e} | dt: {dt*1000:.0f}ms | "
                f"tok/s: {toks_per_sec:.0f} | "
                f"elapsed: {elapsed:.0f}s"
            )
        else:
            g_pos_avg = sum(v for k, v in accum_losses.items() if k.startswith("goodness_pos")) / config.n_layer
            g_neg_avg = sum(v for k, v in accum_losses.items() if k.startswith("goodness_neg")) / config.n_layer
            thresh_str = ""
            if adapt_thresh is not None:
                thresh_str = f" | th: {adapt_thresh.threshold:.2f}"
            print0(
                f"step {step:05d}/{args.num_iterations} | "
                f"lm: {debiased_lm:.3f} | ff: {debiased_ff:.3f} | "
                f"g+: {g_pos_avg:.2f} g-: {g_neg_avg:.2f}{thresh_str} | "
                f"lr: {actual_lr:.2e} | dt: {dt*1000:.0f}ms | "
                f"tok/s: {toks_per_sec:.0f} | "
                f"elapsed: {elapsed:.0f}s"
            )

    if step % 50 == 0:
        log_dict = {"step": step, "train/lm_loss": debiased_lm, "train/ff_avg": debiased_ff, "train/lrm": lrm}
        for k, v in accum_losses.items():
            log_dict[f"train/{k}"] = v
        wandb_run.log(log_dict)

    # --- Checkpoint ---
    if args.save_every > 0 and step > start_step and step % args.save_every == 0:
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/hebbi_{step:06d}.pt"
        ckpt_data = {
            "config": asdict(config),
            "model": model.state_dict(),
            "step": step,
            "dataset": args.dataset,
        }
        if adapt_thresh is not None:
            ckpt_data["adaptive_threshold"] = adapt_thresh.state_dict()
        torch.save(ckpt_data, path)
        print0(f"Saved checkpoint: {path}")

# Final save
os.makedirs("checkpoints", exist_ok=True)
path = "checkpoints/hebbi_final.pt"
final_ckpt = {
    "config": asdict(config),
    "model": model.state_dict(),
    "step": args.num_iterations,
    "dataset": args.dataset,
}
if adapt_thresh is not None:
    final_ckpt["adaptive_threshold"] = adapt_thresh.state_dict()
torch.save(final_ckpt, path)
elapsed = time.time() - t_start
print0(f"\nTraining complete in {elapsed:.0f}s. Final checkpoint: {path}")
print0(f"Total tokens processed: {tokens_processed:,}")
wandb_run.finish()
