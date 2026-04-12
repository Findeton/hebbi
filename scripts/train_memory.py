"""
Memory training for Hebbi — teach blocks to use their Hopfield memory banks.

Loads an SFT checkpoint and trains the memory gate parameters so blocks learn
to integrate retrieved memories. The training loop:

  1. Sample conversation A (context) → forward through model → Hebbian write
     to each block's memory bank
  2. Sample conversation B (target) → forward with banks populated → FF + LM loss
  3. The gate parameter in each bank gets gradients through the FF loss.
     If retrieved memories help, the gate opens. If not, it stays closed.
  4. Clear banks for next step.

This mirrors hippocampal replay → neocortical consolidation:
  - Phase 1 is "forming an episodic memory" (fast Hebbian write)
  - Phase 2 is "using that memory to process new input" (gate learns via FF)

    python -m scripts.train_memory --checkpoint=checkpoints/hebbi_sft_final.pt
    python -m scripts.train_memory --checkpoint=checkpoints/hebbi_sft_final.pt --num-iterations=1000
"""

import os
import time
import json
import argparse
from dataclasses import asdict

import torch
import torch.nn.functional as F

from hebbi.model import DET, DETConfig, norm
from hebbi.local_learning import (
    generate_negatives,
    forward_forward_loss,
    LayerOptimizers,
    get_lr_multiplier,
)
from hebbi.data import get_tokenizer, get_data_loader
from hebbi.common import (
    compute_init, autodetect_device_type, print0, DummyWandb, COMPUTE_DTYPE,
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Memory training for Hebbi")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="SFT checkpoint path")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--dataset", type=str, default="smoltalk",
                    help="dataset for memory training")
# Training
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--num-iterations", type=int, default=1000)
parser.add_argument("--block-lr", type=float, default=3e-4)
parser.add_argument("--head-lr", type=float, default=3e-4)
parser.add_argument("--embed-lr", type=float, default=1e-3)
parser.add_argument("--corruption-rate", type=float, default=0.15)
parser.add_argument("--warmup-steps", type=int, default=50)
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
    wandb_run = wandb_lib.init(project="hebbi-memory", name=args.run, config=vars(args))

# ---------------------------------------------------------------------------
# Load SFT model
# ---------------------------------------------------------------------------
print0(f"Loading checkpoint: {args.checkpoint}")
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
config = DETConfig(**checkpoint["config"])
model = DET(config).to(device)
model.load_state_dict(checkpoint["model"], strict=False)
n_params = sum(p.numel() for p in model.parameters())
n_gate_params = sum(1 for block in model.blocks for _ in [block.memory_bank.gate])
bank_size = config.n_embd * config.n_embd * config.n_layer
print0(f"Config: {json.dumps(asdict(config), indent=2)}")
print0(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
print0(f"Memory banks: {config.n_layer} × {config.n_embd}×{config.n_embd} "
       f"= {bank_size:,} ({bank_size/1e6:.1f}M entries)")
print0(f"Gate params: {n_gate_params} (one per block)")

if args.compile and hasattr(torch, "compile"):
    print0("Compiling model...")
    model = torch.compile(model)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
tokenizer = get_tokenizer()
print0(f"Dataset: {args.dataset} | Vocab: {tokenizer.get_vocab_size()}")

# Two separate data loaders so context and target batches are different
context_loader, _ = get_data_loader(
    args.dataset, tokenizer, args.batch_size, config.sequence_len, device, "train"
)
target_loader, _ = get_data_loader(
    args.dataset, tokenizer, args.batch_size, config.sequence_len, device, "train"
)

# ---------------------------------------------------------------------------
# Memory training step
# ---------------------------------------------------------------------------

def memory_train_step(model, context_ids, target_pos, target_neg,
                      layer_opts, config):
    """
    Memory training step:
      1. Forward context → Hebbian write to banks (no loss, no gradients)
      2. Forward target with populated banks → FF + LM loss (gate gets grads)
      3. Clear banks
    """
    # --- Phase 1: Populate memory banks from context ---
    model.eval()
    with torch.no_grad():
        block_ios = model.forward_and_collect(context_ids)
        model.write_to_banks(block_ios)
    model.train()

    # --- Phase 2: Train on target with banks populated ---
    # This is the standard det_train_step, but now memory_bank.read()
    # returns non-zero values and the gate gets FF gradients
    layer_opts.zero_all()
    losses = {}
    threshold = config.ff_threshold

    # Embed positive and negative
    with torch.no_grad():
        emb_pos = norm(model.wte(target_pos).to(COMPUTE_DTYPE))
        emb_neg = norm(model.wte(target_neg).to(COMPUTE_DTYPE))

    cos_sin = model._get_cos_sin(target_pos.size(1))

    history_pos = [emb_pos]
    history_neg = [emb_neg]

    for i, block in enumerate(model.blocks):
        det_history_pos = [h.detach() for h in history_pos]
        det_history_neg = [h.detach() for h in history_neg]

        out_pos, g_pos = block(det_history_pos[-1], cos_sin,
                               history=det_history_pos)
        out_neg, g_neg = block(det_history_neg[-1], cos_sin,
                               history=det_history_neg)

        ff_loss = forward_forward_loss(g_pos, g_neg, threshold)
        ff_loss.backward()
        torch.nn.utils.clip_grad_norm_(block.parameters(), max_norm=1.0)
        layer_opts.step_block(i)
        layer_opts.block_optimizers[i].zero_grad(set_to_none=True)

        history_pos.append(out_pos.detach())
        history_neg.append(out_neg.detach())
        losses[f"ff_{i}"] = ff_loss.item()
        losses[f"goodness_pos_{i}"] = g_pos.mean().item()
        losses[f"goodness_neg_{i}"] = g_neg.mean().item()
        losses[f"gate_{i}"] = torch.sigmoid(block.memory_bank.gate).item()

    # LM head loss
    x_final = history_pos[-1]
    logits = model.lm_head(norm(x_final)).float()
    targets = target_pos[:, 1:]
    lm_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    emb_for_train = norm(model.wte(target_pos).to(COMPUTE_DTYPE))
    emb_logits = model.lm_head(emb_for_train).float()
    emb_loss = F.cross_entropy(
        emb_logits[:, :-1].reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    total_head_loss = lm_loss + 0.1 * emb_loss
    total_head_loss.backward()
    head_params = []
    for group in layer_opts.head_optimizer.param_groups:
        head_params.extend(group["params"])
    torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
    layer_opts.step_head()

    losses["lm_loss"] = lm_loss.item()
    losses["emb_loss"] = emb_loss.item()

    # --- Phase 3: Clear banks for next step ---
    model.clear_banks()

    return losses


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
layer_opts = LayerOptimizers(model, args.block_lr, args.head_lr, args.embed_lr)

neg_rng = torch.Generator(device="cpu")
neg_rng.manual_seed(1337)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print0(f"\nMemory training for {args.num_iterations} steps...")
print0(f"Each step: context batch → Hebbian write → target batch → FF+LM loss")
print0()

smooth_lm = 0.0
smooth_ff = 0.0
t_start = time.time()

for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations

    # --- Sample ---
    if args.sample_every > 0 and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompt_text = "What is the meaning of life?"
        prompt_ids = (
            [tokenizer.bos_id, tokenizer.user_start_id]
            + tokenizer.encode(prompt_text)
            + [tokenizer.user_end_id, tokenizer.assistant_start_id]
        )
        gen = list(model.generate(prompt_ids, max_tokens=200, temperature=0.8, top_k=40))
        response = tokenizer.decode(gen)
        print0(f"--- sample @ step {step} ---")
        print0(f"User: {prompt_text}")
        print0(f"Assistant: {response[:400]}")
        print0("---")

        # Print gate values
        gate_vals = [torch.sigmoid(b.memory_bank.gate).item()
                     for b in model.blocks]
        print0(f"Gate values: [{', '.join(f'{g:.3f}' for g in gate_vals)}]")
        print0()
        model.train()

    if last_step:
        break

    # --- Training step ---
    t0 = time.time()
    lrm = layer_opts.update_lr(step, args.num_iterations,
                                args.warmup_steps, args.warmdown_ratio)

    # Get context and target batches
    ctx_batch = next(context_loader)
    ctx_ids = ctx_batch[0]  # (B, T)

    tgt_batch = next(target_loader)
    tgt_ids = tgt_batch[0]  # (B, T)

    neg_ids = generate_negatives(tgt_ids, config.vocab_size,
                                 config.corruption_rate, neg_rng)

    losses = memory_train_step(model, ctx_ids, tgt_ids, neg_ids,
                               layer_opts, config)
    dt = time.time() - t0

    # Smoothed losses
    ema = 0.95
    lm_loss = losses["lm_loss"]
    ff_avg = sum(v for k, v in losses.items() if k.startswith("ff_")) / config.n_layer
    smooth_lm = ema * smooth_lm + (1 - ema) * lm_loss
    smooth_ff = ema * smooth_ff + (1 - ema) * ff_avg
    debiased_lm = smooth_lm / (1 - ema ** (step + 1))
    debiased_ff = smooth_ff / (1 - ema ** (step + 1))

    if step % 10 == 0:
        g_pos_avg = sum(v for k, v in losses.items()
                        if k.startswith("goodness_pos")) / config.n_layer
        g_neg_avg = sum(v for k, v in losses.items()
                        if k.startswith("goodness_neg")) / config.n_layer
        gate_avg = sum(v for k, v in losses.items()
                       if k.startswith("gate_")) / config.n_layer
        elapsed = time.time() - t_start
        print0(
            f"step {step:05d}/{args.num_iterations} | "
            f"lm: {debiased_lm:.3f} | ff: {debiased_ff:.3f} | "
            f"g+: {g_pos_avg:.2f} g-: {g_neg_avg:.2f} | "
            f"gate: {gate_avg:.3f} | "
            f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | "
            f"elapsed: {elapsed:.0f}s"
        )

    if step % 50 == 0:
        log_dict = {
            "step": step,
            "memory/lm_loss": debiased_lm,
            "memory/ff_avg": debiased_ff,
            "memory/lrm": lrm,
        }
        for k, v in losses.items():
            log_dict[f"memory/{k}"] = v
        wandb_run.log(log_dict)

    # --- Checkpoint ---
    if args.save_every > 0 and step > 0 and step % args.save_every == 0:
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/hebbi_memory_{step:06d}.pt"
        torch.save({
            "config": asdict(config),
            "model": model.state_dict(),
            "step": step,
            "stage": "memory",
        }, path)
        print0(f"Saved: {path}")

# Final save
os.makedirs("checkpoints", exist_ok=True)
path = "checkpoints/hebbi_memory_final.pt"
torch.save({
    "config": asdict(config),
    "model": model.state_dict(),
    "step": args.num_iterations,
    "stage": "memory",
}, path)
elapsed = time.time() - t_start
print0(f"\nMemory training complete in {elapsed:.0f}s. Checkpoint: {path}")

# Print final gate values
gate_vals = [torch.sigmoid(b.memory_bank.gate).item() for b in model.blocks]
print0(f"Final gate values: [{', '.join(f'{g:.3f}' for g in gate_vals)}]")

wandb_run.finish()
