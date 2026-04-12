"""
Interactive chat with Hebbi model and optional online learning.

    python -m scripts.chat --checkpoint=checkpoints/hebbi_sft_final.pt
    python -m scripts.chat --checkpoint=checkpoints/hebbi_sft_final.pt --online-learning
    python -m scripts.chat --checkpoint=checkpoints/hebbi_sft_final.pt --online-learning --n-mem=16

After each response, type feedback:
  [Enter]     — no feedback (neutral)
  good/g      — positive reinforcement
  bad/b       — negative signal (response treated as negative example)
  quit/q/exit — exit chat
"""

import argparse
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
                    help="enable online FF learning from each conversation turn")
parser.add_argument("--online-lr", type=float, default=3e-4,
                    help="learning rate for online updates")
# GradMem
parser.add_argument("--n-mem", type=int, default=None,
                    help="override n_mem for GradMem (0=disable)")
parser.add_argument("--mem-adapt-steps", type=int, default=5,
                    help="GradMem adaptation steps per turn")
parser.add_argument("--mem-lr", type=float, default=0.01,
                    help="GradMem adaptation learning rate")
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

# Load weights (handle n_mem mismatch gracefully)
state_dict = checkpoint["model"]
if args.n_mem is not None and args.n_mem > 0 and "memory" not in state_dict:
    # Model was trained without memory — initialize fresh
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
    print0("Online learning: ENABLED")
    layer_opts = LayerOptimizers(model, args.online_lr, args.online_lr, args.online_lr)

if config.n_mem > 0:
    print0(f"GradMem: {config.n_mem} prefix tokens (adapt_steps={args.mem_adapt_steps})")

# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
print0("\n--- Hebbi Chat ---")
print0("Type your message. After each response:")
if args.online_learning:
    print0("  [Enter]=neutral  good/g=reinforce  bad/b=negative  quit/q=exit")
else:
    print0("  [Enter]=continue  quit/q=exit")
print0()

conversation_ids = [tokenizer.bos_id]  # running conversation token buffer

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print0("\nGoodbye!")
        break

    if user_input.lower() in ("quit", "q", "exit"):
        print0("Goodbye!")
        break

    if not user_input:
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
    for tok in model.generate(
        conversation_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ):
        # Stop at assistant_end or eos
        if tok in (tokenizer.assistant_end_id, tokenizer.eos_id):
            break
        # Stop at unexpected special tokens
        if tok in (tokenizer.user_start_id, tokenizer.bos_id):
            break
        response_tokens.append(tok)
        print(tokenizer.decode([tok]), end="", flush=True)
    print()  # newline after response

    # Add response to conversation
    conversation_ids.extend(response_tokens)
    conversation_ids.append(tokenizer.assistant_end_id)

    # --- Online learning ---
    if args.online_learning and layer_opts is not None:
        # Get feedback
        try:
            feedback = input("[feedback] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print0("\nGoodbye!")
            break

        if feedback in ("quit", "q", "exit"):
            print0("Goodbye!")
            break

        # Map feedback to reward
        if feedback in ("good", "g", "+", "yes", "y"):
            reward = 1.0
        elif feedback in ("bad", "b", "-", "no", "n"):
            reward = -1.0
        else:
            reward = 0.0  # neutral

        # Run online FF update on the conversation so far
        model.train()
        conv_tensor = torch.tensor(
            [conversation_ids[-config.sequence_len:]],
            dtype=torch.long,
            device=device,
        )
        losses = online_learn_step(model, conv_tensor, layer_opts, config, reward=reward)
        model.eval()

        reward_str = {1.0: "+1 (reinforced)", -1.0: "-1 (corrected)", 0.0: "0 (neutral)"}
        print0(f"  [learned: reward={reward_str.get(reward, reward)}, "
               f"lm={losses['lm_loss']:.3f}]")
    print()
