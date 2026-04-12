"""
SFT (Supervised Fine-Tuning) for Hebbi on conversation data.

Loads a pretrained checkpoint and fine-tunes on conversation datasets
(e.g., SmolTalk) with loss masking — only assistant tokens are trained.

Uses the same per-block Forward-Forward training, but on conversation data.

    python -m scripts.train_sft --checkpoint=checkpoints/hebbi_final.pt --dataset=smoltalk
    python -m scripts.train_sft --checkpoint=checkpoints/hebbi_final.pt --dataset=smoltalk --num-iterations=2000
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
)
from hebbi.data import get_tokenizer, get_data_loader, DATASETS
from hebbi.common import (
    compute_init, autodetect_device_type, print0, DummyWandb, COMPUTE_DTYPE,
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SFT fine-tune Hebbi model")
parser.add_argument("--checkpoint", type=str, required=True, help="pretrained checkpoint path")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--dataset", type=str, default="smoltalk",
                    help="SFT dataset name")
# Training
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--num-iterations", type=int, default=2000)
parser.add_argument("--block-lr", type=float, default=3e-4, help="lower LR for SFT")
parser.add_argument("--head-lr", type=float, default=3e-4)
parser.add_argument("--embed-lr", type=float, default=1e-3)
parser.add_argument("--corruption-rate", type=float, default=0.15)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--save-every", type=int, default=500)
parser.add_argument("--eval-every", type=int, default=100)
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
    wandb_run = wandb_lib.init(project="hebbi-sft", name=args.run, config=vars(args))

# ---------------------------------------------------------------------------
# Load pretrained model
# ---------------------------------------------------------------------------
print0(f"Loading checkpoint: {args.checkpoint}")
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
config = DETConfig(**checkpoint["config"])
model = DET(config).to(device)
model.load_state_dict(checkpoint["model"], strict=False)
n_params = sum(p.numel() for p in model.parameters())
print0(f"Config: {json.dumps(asdict(config), indent=2)}")
print0(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

if args.compile and hasattr(torch, "compile"):
    print0("Compiling model...")
    model = torch.compile(model)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
tokenizer = get_tokenizer()
print0(f"Dataset: {args.dataset} | Vocab: {tokenizer.get_vocab_size()}")

train_loader, _ = get_data_loader(
    args.dataset, tokenizer, args.batch_size, config.sequence_len, device, "train"
)

# ---------------------------------------------------------------------------
# SFT training step with loss masking
# ---------------------------------------------------------------------------

def sft_train_step(model, pos_ids, loss_mask, neg_ids, layer_opts, config):
    """
    SFT training step: same FF per-block training, but LM head loss
    is masked to only compute on assistant tokens.
    """
    layer_opts.zero_all()
    losses = {}
    threshold = config.ff_threshold

    # --- Phase 1: Per-block Forward-Forward (same as pretrain) ---
    with torch.no_grad():
        emb_pos = norm(model.wte(pos_ids).to(COMPUTE_DTYPE))
        emb_neg = norm(model.wte(neg_ids).to(COMPUTE_DTYPE))

    cos_sin = model._get_cos_sin(pos_ids.size(1))

    history_pos = [emb_pos]
    history_neg = [emb_neg]

    for i, block in enumerate(model.blocks):
        det_history_pos = [h.detach() for h in history_pos]
        det_history_neg = [h.detach() for h in history_neg]

        out_pos, g_pos = block(det_history_pos[-1], cos_sin, history=det_history_pos)
        out_neg, g_neg = block(det_history_neg[-1], cos_sin, history=det_history_neg)

        ff_loss = forward_forward_loss(g_pos, g_neg, threshold)
        ff_loss.backward()
        layer_opts.step_block(i)
        layer_opts.block_optimizers[i].zero_grad(set_to_none=True)

        history_pos.append(out_pos.detach())
        history_neg.append(out_neg.detach())
        losses[f"ff_{i}"] = ff_loss.item()

    # --- Phase 2: Masked LM head loss ---
    x_final = history_pos[-1]
    logits = model.lm_head(norm(x_final)).float()
    targets = pos_ids[:, 1:]
    target_mask = loss_mask[:, 1:]  # shift mask to match targets

    # Masked cross-entropy: only compute loss on assistant tokens
    logits_flat = logits[:, :-1].reshape(-1, config.vocab_size)
    targets_flat = targets.reshape(-1)
    mask_flat = target_mask.reshape(-1).float()

    per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    if mask_flat.sum() > 0:
        lm_loss = (per_token_loss * mask_flat).sum() / mask_flat.sum()
    else:
        lm_loss = per_token_loss.mean()

    # Embedding loss (mild, also masked)
    emb_for_train = norm(model.wte(pos_ids).to(COMPUTE_DTYPE))
    emb_logits = model.lm_head(emb_for_train).float()
    emb_flat = emb_logits[:, :-1].reshape(-1, config.vocab_size)
    emb_per_token = F.cross_entropy(emb_flat, targets_flat, reduction="none")
    if mask_flat.sum() > 0:
        emb_loss = (emb_per_token * mask_flat).sum() / mask_flat.sum()
    else:
        emb_loss = emb_per_token.mean()

    total_head_loss = lm_loss + 0.1 * emb_loss
    total_head_loss.backward()
    layer_opts.step_head()

    losses["lm_loss"] = lm_loss.item()
    losses["emb_loss"] = emb_loss.item()
    losses["mask_ratio"] = mask_flat.mean().item()
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
print0(f"\nSFT training for {args.num_iterations} steps...")
print0()

smooth_lm = 0.0
t_start = time.time()

for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations

    # --- Sample ---
    if args.sample_every > 0 and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        # Generate a chat-style response
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
        model.train()

    if last_step:
        break

    # --- Training step ---
    t0 = time.time()
    lrm = layer_opts.update_lr(step, args.num_iterations, args.warmup_steps, args.warmdown_ratio)

    x, y, loss_mask = next(train_loader)
    neg_ids = generate_negatives(x, config.vocab_size, config.corruption_rate, neg_rng)

    losses = sft_train_step(model, x, loss_mask, neg_ids, layer_opts, config)
    dt = time.time() - t0

    # Smoothed loss
    ema = 0.95
    lm_loss = losses["lm_loss"]
    smooth_lm = ema * smooth_lm + (1 - ema) * lm_loss
    debiased_lm = smooth_lm / (1 - ema ** (step + 1))

    if step % 10 == 0:
        elapsed = time.time() - t_start
        print0(
            f"step {step:05d}/{args.num_iterations} | "
            f"lm: {debiased_lm:.3f} | "
            f"mask: {losses['mask_ratio']:.2f} | "
            f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | "
            f"elapsed: {elapsed:.0f}s"
        )

    if step % 50 == 0:
        log_dict = {"step": step, "sft/lm_loss": debiased_lm, "sft/lrm": lrm}
        for k, v in losses.items():
            log_dict[f"sft/{k}"] = v
        wandb_run.log(log_dict)

    # --- Checkpoint ---
    if args.save_every > 0 and step > 0 and step % args.save_every == 0:
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/hebbi_sft_{step:06d}.pt"
        torch.save({
            "config": asdict(config),
            "model": model.state_dict(),
            "step": step,
            "stage": "sft",
        }, path)
        print0(f"Saved: {path}")

# Final save
os.makedirs("checkpoints", exist_ok=True)
path = "checkpoints/hebbi_sft_final.pt"
torch.save({
    "config": asdict(config),
    "model": model.state_dict(),
    "step": args.num_iterations,
    "stage": "sft",
}, path)
elapsed = time.time() - t_start
print0(f"\nSFT complete in {elapsed:.0f}s. Checkpoint: {path}")
wandb_run.finish()
