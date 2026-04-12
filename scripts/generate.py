"""
Generate text from a trained Hebbi model.

    python -m scripts.generate --checkpoint checkpoints/hebbi_final.pt
    python -m scripts.generate --checkpoint checkpoints/hebbi_final.pt --prompt "Once upon a time"
    python -m scripts.generate --checkpoint checkpoints/hebbi_final.pt --prompt "To be or not" --energy-steps 3
"""

import argparse
import torch

from hebbi.model import DET, DETConfig
from hebbi.data import get_shakespeare, CharDataset, get_tokenizer
from hebbi.common import compute_init, autodetect_device_type


parser = argparse.ArgumentParser(description="Generate text from Hebbi model")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--max-tokens", type=int, default=500)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=40)
parser.add_argument("--energy-steps", type=int, default=None, help="override thinking iterations")
parser.add_argument("--device-type", type=str, default="")
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = compute_init(device_type)

# Load checkpoint
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
config = DETConfig(**checkpoint["config"])
if args.energy_steps is not None:
    config.energy_steps = args.energy_steps

model = DET(config).to(device)
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()

# Detect if this is a char-level or BPE model
is_char = config.vocab_size < 256

if is_char:
    dataset = CharDataset(get_shakespeare())
    prompt = args.prompt or "ROMEO:"
    tokens = dataset.encode(prompt)
    print(prompt, end="", flush=True)
    for tok in model.generate(tokens, args.max_tokens, args.temperature, args.top_k):
        print(dataset.decode([tok]), end="", flush=True)
else:
    tokenizer = get_tokenizer()
    prompt = args.prompt or "Once upon a time"
    tokens = tokenizer.encode(prompt)
    print(prompt, end="", flush=True)
    for tok in model.generate(tokens, args.max_tokens, args.temperature, args.top_k):
        print(tokenizer.decode([tok]), end="", flush=True)

print()
