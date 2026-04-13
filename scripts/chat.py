"""
Interactive chat with Hebbi model and two-speed learning.

Fast path (every turn): reward-modulated Hebbian writes to memory banks.
  Instant, no gradients. The bank learns what patterns to retrieve (good)
  and what to suppress (bad) via dopamine-like modulation.

Slow path (/sleep): pure Hebbian consolidation via dreaming.
  Generates "dream" sequences from the currently-populated memory banks
  (the banks bias the sampling toward stored attractors), then runs
  Forward-Forward weight updates on those dreams. The bank's contents
  thereby get baked into permanent model weights — without ever storing
  raw conversation text. Like hippocampal → cortical replay during sleep.

    python -m scripts.chat --checkpoint=checkpoints/hebbi_memory_final.pt --online-learning
    python -m scripts.chat --checkpoint=checkpoints/hebbi_memory_final.pt --online-learning --n-mem=16

Commands:
  [Enter]     — no feedback (neutral, mild Hebbian write)
  good/g      — positive (strong Hebbian write)
  bad/b       — negative (anti-Hebbian write, suppresses pattern)
  /sleep      — consolidate: dream from banks and FF-train on dreams
  quit/q/exit — save and exit
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from hebbi.model import DET, DETConfig, norm
from hebbi.data import get_tokenizer
from hebbi.local_learning import (
    adapt_memory,
    LayerOptimizers,
    generate_negatives,
    forward_forward_loss,
)
from hebbi.common import compute_init, autodetect_device_type, print0, COMPUTE_DTYPE

parser = argparse.ArgumentParser(description="Chat with Hebbi model")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--max-tokens", type=int, default=512, help="max response length")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=40)
# Online learning
parser.add_argument("--online-learning", action="store_true",
                    help="enable online learning (Hebbian + /sleep FF consolidation)")
parser.add_argument("--online-lr", type=float, default=3e-4,
                    help="learning rate for FF updates during /sleep")
# GradMem
parser.add_argument("--n-mem", type=int, default=None,
                    help="override n_mem for GradMem (0=disable)")
parser.add_argument("--mem-adapt-steps", type=int, default=5,
                    help="GradMem adaptation steps per turn")
parser.add_argument("--mem-lr", type=float, default=0.01,
                    help="GradMem adaptation learning rate")
# Sleep / consolidation
parser.add_argument("--sleep-dreams", type=int, default=16,
                    help="number of dream sequences generated per /sleep call")
parser.add_argument("--sleep-epochs", type=int, default=2,
                    help="number of passes over each generated dream batch")
parser.add_argument("--dream-length", type=int, default=128,
                    help="max tokens per dream sequence")
parser.add_argument("--dream-temperature", type=float, default=1.0,
                    help="sampling temperature for dream generation")
parser.add_argument("--sleep-bank-decay", type=float, default=0.3,
                    help="bank W is multiplied by this after /sleep "
                         "(0=clear, 1=keep — default 0.3 keeps weakened residual)")
parser.add_argument("--sleep-distill-weight", type=float, default=0.5,
                    help="weight of distillation loss vs FF loss during /sleep "
                         "(0=pure FF on dreams, higher=stronger bank→weight transfer)")
parser.add_argument("--backprop", action="store_true",
                    help="use backprop for /sleep consolidation instead of FF")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = compute_init(device_type)

# Load model
print0(f"Loading checkpoint: {args.checkpoint}")
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
config = DETConfig(**checkpoint["config"])

# Override n_mem if requested
if args.n_mem is not None:
    config.n_mem = args.n_mem

model = DET(config).to(device)

# Load weights (handle missing memory bank keys gracefully)
state_dict = checkpoint["model"]
if args.n_mem is not None and args.n_mem > 0 and "memory" not in state_dict:
    print0(f"Initializing {args.n_mem} GradMem tokens (not in checkpoint)")
model.load_state_dict(state_dict, strict=False)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print0(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

# Tokenizer
tokenizer = get_tokenizer()

# Online learning setup
layer_opts = None
if args.online_learning:
    print0("Online learning: ENABLED (Hebbian fast path + /sleep consolidation)")
    layer_opts = LayerOptimizers(model, args.online_lr, args.online_lr, args.online_lr)

# RNG for FF negatives during /sleep (CPU: MPS doesn't support torch.Generator)
neg_rng = torch.Generator(device="cpu")
neg_rng.manual_seed(1337)

if config.n_mem > 0:
    print0(f"GradMem: {config.n_mem} prefix tokens (adapt_steps={args.mem_adapt_steps})")

# ---------------------------------------------------------------------------
# Sleep consolidation — pure Hebbian dream replay
# ---------------------------------------------------------------------------

def dream_train_step(model, dream_ids, neg_ids, layer_opts, config,
                     distill_weight=0.5, lr_scale=0.5):
    """
    One /sleep training step on a dreamed sequence.

    Combines TWO local objectives in a single block-by-block pass:

      (1) Forward-Forward on the dream — teaches the model that dream
          sequences are "positive" (high goodness) vs their corrupted
          version (negative, low goodness). Same mechanism as waking FF.

      (2) Bank→Weight Distillation — for each block, run the forward
          TWICE: once with its memory bank contribution active (teacher),
          once with it disabled (student). MSE between the two outputs
          drives the student to produce what the teacher produces,
          absorbing the bank's contribution into the block's base weights.

    Both losses are computed PER BLOCK and backward()ed together, so no
    gradient ever flows between blocks — FF locality is preserved. The
    distillation term has a clean termination condition: when it reaches
    zero, the base weights can reproduce the bank's retrieval output on
    their own, and consolidation for that block is complete.

    Args:
        model: DET
        dream_ids: (1, T) dream sequence token ids
        neg_ids: (1, T) corrupted dream for FF negative
        layer_opts: LayerOptimizers
        config: DETConfig
        distill_weight: coefficient on MSE distillation loss (0 disables)
        lr_scale: scale factor on LRs for this step (dreams are less
                  trusted than waking data — default 0.5)
    """
    layer_opts.zero_all()
    losses = {}
    threshold = config.ff_threshold
    n_layer = len(model.blocks)

    # Temporarily scale learning rates (dreams are half-trusted)
    all_opts = layer_opts.block_optimizers + [layer_opts.head_optimizer]
    saved_lrs = []
    for opt in all_opts:
        group_lrs = []
        for g in opt.param_groups:
            group_lrs.append(g["lr"])
            g["lr"] = g["lr"] * lr_scale
        saved_lrs.append(group_lrs)

    try:
        with torch.no_grad():
            emb_pos = norm(model.wte(dream_ids).to(COMPUTE_DTYPE))
            emb_neg = norm(model.wte(neg_ids).to(COMPUTE_DTYPE))

        cos_sin = model._get_cos_sin(dream_ids.size(1))

        history_pos = [emb_pos]
        history_neg = [emb_neg]

        total_ff = 0.0
        total_distill = 0.0

        for i, block in enumerate(model.blocks):
            det_hist_pos = [h.detach() for h in history_pos]
            det_hist_neg = [h.detach() for h in history_neg]

            # --- Teacher pass: banks ON (normal forward) ---
            # This is also the "positive" FF example — its goodness should
            # be above threshold.
            out_pos_on, g_pos = block(det_hist_pos[-1], cos_sin,
                                      history=det_hist_pos)
            out_neg_on, g_neg = block(det_hist_neg[-1], cos_sin,
                                      history=det_hist_neg)

            ff_loss = forward_forward_loss(g_pos, g_neg, threshold)

            # --- Student pass: THIS block's bank temporarily OFF ---
            # HopfieldMemoryBank.read() short-circuits to zeros when
            # n_writes == 0, so zeroing it cleanly bypasses the bank
            # contribution for this one forward call.
            if distill_weight > 0 and block.memory_bank.n_writes.item() > 0:
                saved_n_writes = block.memory_bank.n_writes.clone()
                block.memory_bank.n_writes.zero_()
                try:
                    out_pos_off, _ = block(det_hist_pos[-1], cos_sin,
                                           history=det_hist_pos)
                finally:
                    block.memory_bank.n_writes.copy_(saved_n_writes)

                # Distillation: student (banks-off) matches teacher (banks-on).
                # .detach() on the target keeps this from backpropping into
                # the teacher path — gradients flow only through the student
                # forward, which updates the same block's parameters.
                distill_loss = F.mse_loss(out_pos_off, out_pos_on.detach())
            else:
                distill_loss = torch.zeros((), device=dream_ids.device,
                                           dtype=out_pos_on.dtype)

            total = ff_loss + distill_weight * distill_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(block.parameters(), max_norm=1.0)
            layer_opts.step_block(i)
            layer_opts.block_optimizers[i].zero_grad(set_to_none=True)

            # History flows forward with the bank-influenced output, so
            # later blocks see the same context the teacher produced.
            history_pos.append(out_pos_on.detach())
            history_neg.append(out_neg_on.detach())

            ff_val = ff_loss.item()
            distill_val = distill_loss.item() if distill_weight > 0 else 0.0
            total_ff += ff_val
            total_distill += distill_val
            losses[f"ff_{i}"] = ff_val
            losses[f"distill_{i}"] = distill_val

        # --- LM head on final block output (detached) ---
        x_final = history_pos[-1]
        logits = model.lm_head(norm(x_final)).float()
        targets = dream_ids[:, 1:]
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            targets.reshape(-1),
        )
        lm_loss.backward()
        head_params = []
        for group in layer_opts.head_optimizer.param_groups:
            head_params.extend(group["params"])
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
        layer_opts.step_head()

        losses["lm_loss"] = lm_loss.item()
        losses["ff_avg"] = total_ff / n_layer
        losses["distill_avg"] = total_distill / n_layer
    finally:
        # Restore learning rates
        for opt, group_lrs in zip(all_opts, saved_lrs):
            for g, lr in zip(opt.param_groups, group_lrs):
                g["lr"] = lr

    return losses


def dream_backprop_step(model, dream_ids, layer_opts, config, lr_scale=0.5):
    """
    Backprop /sleep step: full forward on dream, backprop LM loss through all blocks.
    """
    layer_opts.zero_all()
    losses = {}

    # Temporarily scale learning rates
    all_opts = layer_opts.block_optimizers + [layer_opts.head_optimizer]
    saved_lrs = []
    for opt in all_opts:
        group_lrs = []
        for g in opt.param_groups:
            group_lrs.append(g["lr"])
            g["lr"] = g["lr"] * lr_scale
        saved_lrs.append(group_lrs)

    try:
        x = norm(model.wte(dream_ids).to(COMPUTE_DTYPE))
        cos_sin = model._get_cos_sin(dream_ids.size(1))
        x, _ = model.forward_blocks(x, cos_sin, return_goodness=False)

        logits = model.lm_head(norm(x)).float()
        targets = dream_ids[:, 1:]
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            targets.reshape(-1),
        )
        lm_loss.backward()
        all_params = list(model.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        for i in range(len(model.blocks)):
            layer_opts.step_block(i)
        layer_opts.step_head()

        losses["lm_loss"] = lm_loss.item()
        losses["ff_avg"] = 0.0
        losses["distill_avg"] = 0.0
    finally:
        for opt, group_lrs in zip(all_opts, saved_lrs):
            for g, lr in zip(opt.param_groups, group_lrs):
                g["lr"] = lr

    return losses


def run_sleep_hebbian(model, layer_opts, config, tokenizer, neg_rng,
                     n_dreams=16, n_epochs=2,
                     dream_length=128, dream_temperature=1.0,
                     distill_weight=0.5, bank_decay_after=0.3,
                     use_backprop=False):
    """
    Memory consolidation via bank-driven dreams (pure Hebbian replay).

    No stored text logs: the memory banks ARE the memory. The procedure:

      1. INSPECT banks — warn if empty or gates mostly closed.
      2. For each of n_dreams:
         a. Generate a token sequence from the bos token with banks populated.
            Each block's bank retrieval `sigmoid(gate) * W @ x` biases the
            forward pass toward stored attractors — the dream reflects what
            the banks encode. Higher temperature = more exploration.
         b. Run dream_train_step on the dream. This combines:
              • Forward-Forward loss: dream is positive, corrupted dream is
                negative. Makes the dream sequences high-goodness.
              • Bank→weight distillation: per block, MSE between the output
                with banks on (teacher) and banks off (student). Transfers
                bank contribution into base weights.
            Both are computed per-block with local .backward() — no
            cross-block gradient flow, FF locality preserved.
      3. After all dreams, decay banks by `bank_decay_after`:
         - 0.0 = clear banks (full reset)
         - 0.3 = keep a weakened residual (default — preserves continuity)
         - 1.0 = leave banks untouched
         The strong signals have been consolidated into weights (distillation
         loss approaches zero as this happens). The weakened residual is
         kept as working memory for the next conversation.

    Biological analogy: hippocampal sharp-wave ripples during NREM sleep
    reactivate stored patterns, which cascade through cortex and drive
    synaptic consolidation. Hebbi's bank is the hippocampus (fast Hebbian);
    the FF-trained block weights are the neocortex (slow consolidation).
    Distillation's "banks off matches banks on" objective is the explicit
    handoff from episodic to semantic memory.

    Returns: number of dream-train steps executed.
    """
    # --- 1. Inspect bank state ---
    n_layer = len(model.blocks)
    total_writes = sum(b.memory_bank.n_writes.item() for b in model.blocks)
    gate_vals = [torch.sigmoid(b.memory_bank.gate).item() for b in model.blocks]
    avg_gate = sum(gate_vals) / n_layer

    if total_writes == 0:
        print0("  Banks are empty — no memories to consolidate.")
        print0("  (Hint: chat a few turns first; each turn writes to banks.)")
        return 0

    if avg_gate < 0.05:
        print0(f"  Warning: gates are mostly closed (avg={avg_gate:.3f}).")
        print0("  Dreams will carry weak bank signal. Consider running stage 4")
        print0("  memory training to open the gates.")

    print0(f"  Banks: {total_writes} writes across {n_layer} blocks, "
           f"avg gate={avg_gate:.3f}")
    print0(f"  Generating {n_dreams} dreams × {n_epochs} epochs "
           f"(len={dream_length}, temp={dream_temperature})...")

    # --- 2. Dream generation + FF + distillation ---
    t0 = time.time()
    total_ff = 0.0
    total_distill = 0.0
    total_lm = 0.0
    total_steps = 0
    dreams_skipped = 0

    # Use a few different seeds so dreams aren't identical
    dream_seeds = [(hash((epoch, d, total_writes)) & 0x7FFFFFFF)
                   for epoch in range(n_epochs) for d in range(n_dreams)]
    seed_idx = 0

    for epoch in range(n_epochs):
        for dream_idx in range(n_dreams):
            # --- 2a. Generate dream from populated banks ---
            model.eval()
            with torch.no_grad():
                generated = list(model.generate(
                    [tokenizer.bos_id],
                    max_tokens=dream_length,
                    temperature=dream_temperature,
                    top_k=40,
                    seed=dream_seeds[seed_idx],
                ))
            seed_idx += 1

            # Filter: reject degenerate dreams (too short, collapsed, etc.)
            if len(generated) < 8:
                dreams_skipped += 1
                continue
            if len(set(generated)) < 3:
                dreams_skipped += 1
                continue

            # Full dream sequence (include the seed bos)
            dream_ids = [tokenizer.bos_id] + generated
            dream_ids = dream_ids[-config.sequence_len:]
            dream_tensor = torch.tensor(
                [dream_ids], dtype=torch.long, device=device
            )
            neg_tensor = generate_negatives(
                dream_tensor, config.vocab_size,
                config.corruption_rate, neg_rng,
            )

            # --- 2b. Train on the dream ---
            if use_backprop:
                losses = dream_backprop_step(
                    model, dream_tensor, layer_opts, config, lr_scale=0.5,
                )
            else:
                losses = dream_train_step(
                    model, dream_tensor, neg_tensor, layer_opts, config,
                    distill_weight=distill_weight, lr_scale=0.5,
                )
            total_lm += losses["lm_loss"]
            total_ff += losses["ff_avg"]
            total_distill += losses["distill_avg"]
            total_steps += 1

    model.eval()

    if total_steps == 0:
        print0(f"  All {dreams_skipped} dreams were degenerate — no training.")
        print0("  (Gates may be too closed, or the model may be undertrained.)")
        return 0

    avg_lm = total_lm / total_steps
    avg_ff = total_ff / total_steps
    avg_distill = total_distill / total_steps
    elapsed = time.time() - t0

    # --- 3. Decay banks (don't clear) ---
    # The strong signals have been baked into weights; we keep a weakened
    # residual of the bank so the next conversation has some continuity.
    if bank_decay_after <= 0.0:
        model.clear_banks()
        bank_status = "cleared"
    elif bank_decay_after >= 1.0:
        bank_status = "untouched"
    else:
        with torch.no_grad():
            for block in model.blocks:
                block.memory_bank.W.mul_(bank_decay_after)
        bank_status = f"decayed to {bank_decay_after:.0%}"

    new_gate_vals = [torch.sigmoid(b.memory_bank.gate).item()
                     for b in model.blocks]
    print0(f"  {total_steps} dream-train steps in {elapsed:.1f}s "
           f"({dreams_skipped} skipped)")
    print0(f"  avg lm={avg_lm:.3f} | avg ff={avg_ff:.3f} | "
           f"avg distill={avg_distill:.4f}")
    if distill_weight > 0:
        # distill loss → 0 means the bank's contribution has been fully
        # absorbed into weights. User can watch this trend across sleeps.
        if avg_distill < 1e-4:
            print0("  Distillation converged — banks fully consolidated.")
    print0(f"  Gate values: [{', '.join(f'{g:.3f}' for g in new_gate_vals)}]")
    print0(f"  Banks: {bank_status}")

    return total_steps


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
conversation_ids = [tokenizer.bos_id]
n_online_updates = 0


def save_on_exit():
    """Save model checkpoint if any learning occurred."""
    if n_online_updates == 0:
        return
    save_path = Path(args.checkpoint).with_suffix(".chat.pt")
    print0(f"\nSaving model after {n_online_updates} updates → {save_path}")
    ckpt = {
        "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
        "model": model.state_dict(),
        "step": checkpoint.get("step", 0),
        "chat_updates": n_online_updates,
    }
    torch.save(ckpt, save_path)
    print0(f"Saved ({save_path.stat().st_size / 1e6:.1f} MB). "
           f"Resume with: --checkpoint={save_path}")


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
print0("\n--- Hebbi Chat ---")
print0("Type your message. After each response:")
if args.online_learning:
    print0("  [Enter]=neutral  good/g=positive  bad/b=negative")
    print0("  /sleep=consolidate memories  quit/q=exit")
else:
    print0("  [Enter]=continue  quit/q=exit")
print0()

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        save_on_exit()
        print0("\nGoodbye!")
        break

    if user_input.lower() in ("quit", "q", "exit"):
        save_on_exit()
        print0("Goodbye!")
        break

    if not user_input:
        continue

    # --- /sleep command ---
    if user_input.lower() in ("/sleep", "sleep"):
        if not args.online_learning or layer_opts is None:
            print0("Use --online-learning to enable /sleep.")
            print0()
            continue
        print0("Entering sleep mode — dreaming from memory banks...")
        n_steps = run_sleep_hebbian(
            model, layer_opts, config, tokenizer, neg_rng,
            n_dreams=args.sleep_dreams,
            n_epochs=args.sleep_epochs,
            dream_length=args.dream_length,
            dream_temperature=args.dream_temperature,
            distill_weight=args.sleep_distill_weight,
            bank_decay_after=args.sleep_bank_decay,
            use_backprop=args.backprop,
        )
        if n_steps:
            n_online_updates += n_steps
        print0()
        continue

    # Build user turn tokens
    user_tokens = (
        [tokenizer.user_start_id]
        + tokenizer.encode(user_input)
        + [tokenizer.user_end_id, tokenizer.assistant_start_id]
    )
    conversation_ids.extend(user_tokens)

    # Truncate to fit sequence length
    max_context = config.sequence_len - args.max_tokens
    if len(conversation_ids) > max_context:
        conversation_ids = [tokenizer.bos_id] + conversation_ids[-(max_context - 1):]

    # GradMem: adapt memory to current conversation context
    if config.n_mem > 0 and len(conversation_ids) > 4:
        ctx = torch.tensor([conversation_ids], dtype=torch.long, device=device)
        adapt_memory(model, ctx, n_steps=args.mem_adapt_steps, lr=args.mem_lr)

    # Generate response
    print("Assistant: ", end="", flush=True)
    response_tokens = []
    newline_count = 0
    for tok in model.generate(
        conversation_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ):
        if tok in (tokenizer.assistant_end_id, tokenizer.eos_id):
            break
        if tok in (tokenizer.user_start_id, tokenizer.bos_id):
            break
        decoded = tokenizer.decode([tok])
        response_tokens.append(tok)
        print(decoded, end="", flush=True)
        if "\n" in decoded:
            newline_count += 1
        else:
            newline_count = 0
        if newline_count >= 3:
            break
    print()

    # Add response to conversation
    conversation_ids.extend(response_tokens)
    conversation_ids.append(tokenizer.assistant_end_id)

    # --- Fast path: reward-modulated Hebbian learning ---
    if args.online_learning and layer_opts is not None:
        # Get feedback
        try:
            feedback = input("[feedback] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            save_on_exit()
            print0("\nGoodbye!")
            break

        if feedback in ("quit", "q", "exit"):
            save_on_exit()
            print0("Goodbye!")
            break

        if feedback in ("/sleep", "sleep"):
            print0("Entering sleep mode — dreaming from memory banks...")
            n_steps = run_sleep_hebbian(
                model, layer_opts, config, tokenizer, neg_rng,
                n_dreams=args.sleep_dreams,
                n_epochs=args.sleep_epochs,
                dream_length=args.dream_length,
                dream_temperature=args.dream_temperature,
                distill_weight=args.sleep_distill_weight,
                bank_decay_after=args.sleep_bank_decay,
                use_backprop=args.backprop,
            )
            if n_steps:
                n_online_updates += n_steps
            print0()
            continue

        # Map feedback to reward
        if feedback in ("good", "g", "+", "yes", "y"):
            reward = 1.0
        elif feedback in ("bad", "b", "-", "no", "n"):
            reward = -1.0
        else:
            reward = 0.0

        # Fast path: reward-modulated Hebbian write to memory banks
        # No gradients, no backward pass — just one outer product per block
        conv_tensor = torch.tensor(
            [conversation_ids[-config.sequence_len:]],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            block_ios = model.forward_and_collect(conv_tensor)
            model.write_to_banks(block_ios, reward=reward)

        n_online_updates += 1

        strength = 0.3 + 0.7 * reward
        reward_labels = {1.0: "+1 (LTP)", -1.0: "-1 (LTD)", 0.0: "0 (mild)"}
        n_writes = model.blocks[0].memory_bank.n_writes.item()
        print0(f"  [hebbian: reward={reward_labels.get(reward, reward)}, "
               f"strength={strength:.1f}, "
               f"bank={n_writes} memories]")
    else:
        # No online learning — still write to banks with neutral reward
        conv_tensor = torch.tensor(
            [conversation_ids[-config.sequence_len:]],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            block_ios = model.forward_and_collect(conv_tensor)
            model.write_to_banks(block_ios, reward=0.0)

    print()
