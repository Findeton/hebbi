"""
Interactive chat with Hebbi model and two-speed learning.

Fast path (every turn): reward-modulated Hebbian writes to memory banks.
  Instant, no gradients. The bank learns what patterns to retrieve (good)
  and what to suppress (bad) via dopamine-like modulation.

Slow path (/sleep): FF weight replay consolidation.
  Replays logged episodes through full Forward-Forward training, baking
  episodic memories into permanent model weights. Like sleep consolidation.

    python -m scripts.chat --checkpoint=checkpoints/hebbi_memory_final.pt --online-learning
    python -m scripts.chat --checkpoint=checkpoints/hebbi_memory_final.pt --online-learning --n-mem=16

Commands:
  [Enter]     — no feedback (neutral, mild Hebbian write)
  good/g      — positive (strong Hebbian write)
  bad/b       — negative (anti-Hebbian write, suppresses pattern)
  /sleep      — consolidate: replay past episodes through FF weight updates
  quit/q/exit — save and exit
"""

import argparse
import json
import time
from pathlib import Path

import torch

from hebbi.model import DET, DETConfig
from hebbi.data import get_tokenizer
from hebbi.local_learning import (
    online_learn_step,
    adapt_memory,
    LayerOptimizers,
)
from hebbi.common import compute_init, autodetect_device_type, print0

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
parser.add_argument("--sleep-epochs", type=int, default=3,
                    help="number of replay passes over logged episodes during /sleep")
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

if config.n_mem > 0:
    print0(f"GradMem: {config.n_mem} prefix tokens (adapt_steps={args.mem_adapt_steps})")

# ---------------------------------------------------------------------------
# Episodic memory — conversation log for /sleep consolidation
# ---------------------------------------------------------------------------
EPISODE_LOG = Path(args.checkpoint).with_suffix(".episodes.jsonl")

episodes = []
if EPISODE_LOG.exists():
    for line in EPISODE_LOG.read_text().strip().split("\n"):
        if line.strip():
            episodes.append(json.loads(line))
    print0(f"Loaded {len(episodes)} past episodes from {EPISODE_LOG.name}")


def log_episode(token_ids, reward):
    """Append one conversation turn to the episode log."""
    entry = {
        "tokens": token_ids[-config.sequence_len:],
        "reward": reward,
        "timestamp": time.time(),
    }
    episodes.append(entry)
    with open(EPISODE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_sleep(model, layer_opts, config, episodes, n_epochs=3):
    """
    Memory consolidation via FF replay (slow path).

    Replays past episodes through full Forward-Forward training:
    - Each episode is first written to memory banks (Hebbian context)
    - Then used for FF weight updates with the original reward
    - Recent episodes are replayed more often (recency weighting)

    This bakes episodic (Hebbian) memories into permanent (weight) memories.
    Like hippocampal replay during biological sleep.
    """
    if not episodes:
        print0("  No episodes to replay.")
        return 0

    import random
    print0(f"  Replaying {len(episodes)} episodes × {n_epochs} epochs...")

    # Recency weighting: episode i gets weight (i+1)/N
    n = len(episodes)
    weights = [(i + 1) / n for i in range(n)]

    model.train()
    total_steps = 0
    total_lm = 0.0

    for epoch in range(n_epochs):
        # Sample with recency bias
        indices = random.choices(range(n), weights=weights, k=n)

        for idx in indices:
            ep = episodes[idx]
            token_ids = ep["tokens"]
            reward = ep["reward"]

            if len(token_ids) < 4:
                continue

            conv_tensor = torch.tensor(
                [token_ids], dtype=torch.long, device=device
            )

            # Populate banks with Hebbian context from this episode
            with torch.no_grad():
                block_ios = model.forward_and_collect(conv_tensor)
                model.write_to_banks(block_ios, reward=reward)

            # FF weight update (slow path — consolidation)
            losses = online_learn_step(
                model, conv_tensor, layer_opts, config, reward=reward
            )
            total_lm += losses["lm_loss"]
            total_steps += 1

    model.eval()
    avg_lm = total_lm / max(total_steps, 1)

    # Print gate values after consolidation
    gate_vals = [torch.sigmoid(b.memory_bank.gate).item()
                 for b in model.blocks]
    print0(f"  {n_epochs} epochs, {total_steps} replay steps, avg lm={avg_lm:.3f}")
    print0(f"  Gate values: [{', '.join(f'{g:.3f}' for g in gate_vals)}]")

    # Clear banks after sleep (fresh start for next conversation)
    model.clear_banks()

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
        print0("Entering sleep mode — consolidating memories...")
        n_steps = run_sleep(model, layer_opts, config, episodes,
                            n_epochs=args.sleep_epochs)
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
            print0("Entering sleep mode — consolidating memories...")
            n_steps = run_sleep(model, layer_opts, config, episodes,
                                n_epochs=args.sleep_epochs)
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

        # Log episode for future /sleep consolidation
        log_episode(conversation_ids, reward)
        n_online_updates += 1

        strength = 0.3 + 0.7 * reward
        reward_labels = {1.0: "+1 (LTP)", -1.0: "-1 (LTD)", 0.0: "0 (mild)"}
        n_writes = model.blocks[0].memory_bank.n_writes.item()
        print0(f"  [hebbian: reward={reward_labels.get(reward, reward)}, "
               f"strength={strength:.1f}, "
               f"bank={n_writes} memories, "
               f"episodes={len(episodes)}]")
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
