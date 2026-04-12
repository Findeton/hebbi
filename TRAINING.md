# Hebbi Training Guide

## Overview

Hebbi uses Forward-Forward local learning — each block trains independently with its own optimizer, no gradient flows between blocks. This guide covers the full training pipeline from pretraining to chat with online learning.

## Hardware Requirements

| Model Size | Depth | n_embd | Params | GPU Memory | Training Time |
|-----------|-------|--------|--------|------------|--------------|
| Tiny      | 3     | 192    | ~5M    | CPU/MPS OK | Minutes      |
| Small     | 6     | 384    | ~20M   | ~4 GB      | ~1 hour      |
| Medium    | 12    | 768    | ~100M  | ~16 GB     | ~4-8 hours   |

Times are approximate for single GPU with default hyperparameters.

## Installation

```bash
git clone https://github.com/Findeton/hebbi.git
cd hebbi
pip install -e .

# Optional: wandb for logging
pip install wandb
```

## Quick Start (CPU, Shakespeare)

Validate everything works with the small character-level model:

```bash
python -m scripts.train \
    --dataset=shakespeare \
    --depth=3 --n-embd=64 --seq-len=64 \
    --batch-size=4 --num-iterations=100
```

## One-Command Pipeline (Resumable)

For the full TinyStories → ClimbMix → SmolTalk SFT run, use
[scripts/run_pipeline.py](scripts/run_pipeline.py). It runs all three stages in
sequence, each in its own directory under `/workspace/runs/`, and is fully
resumable — if the pod restarts or you Ctrl+C, just re-run the same command.

It will:
1. Skip stages already marked complete in `pipeline_state.json`
2. For an in-progress stage, resume from the latest intermediate checkpoint
3. For a fresh stage, auto-bootstrap weights from the previous stage's
   `hebbi_final.pt` (with the step counter reset to 0 so the LR schedule
   starts clean)

### Bootstrap on RunPod (one time)

```bash
cd /workspace
git clone https://github.com/Findeton/hebbi.git
cd hebbi
pip install -e .
```

### Start the pipeline

Run it in `tmux` so it survives SSH drops:

```bash
tmux new -s train
cd /workspace/hebbi
export HF_HOME=/workspace/hf_cache   # persist HF cache on the volume
python scripts/run_pipeline.py
# Ctrl+B then D to detach; reattach with: tmux attach -t train
```

### Resume after interruption

Just re-run the same command — no flags needed:

```bash
cd /workspace/hebbi
export HF_HOME=/workspace/hf_cache
python scripts/run_pipeline.py
```

### Final outputs

- `/workspace/runs/stage_tinystories/checkpoints/hebbi_final.pt`
- `/workspace/runs/stage_climbmix/checkpoints/hebbi_final.pt`
- `/workspace/runs/stage_sft/checkpoints/hebbi_sft_final.pt`
- `/workspace/runs/stage_memory/checkpoints/hebbi_memory_final.pt` ← the chatbot (with trained memory gates)

### GPU memory tuning

Defaults target a 24GB card (RTX A5000 / 3090 / 4090):
`--batch-size=8 --grad-accum=6` for TinyStories, `--grad-accum=12` for ClimbMix.
Forward-Forward keeps per-block activations for local learning, so memory
scales faster than standard backprop training — if you OOM, halve
`batch-size` and double `grad-accum` (effective batch stays the same). On an
H100/A100 (80GB) you can push `batch-size=24` for faster throughput.

---

## Training Pipeline

### Stage 1: Pretrain on TinyStories

Small dataset (~500MB) for validating the architecture at scale with BPE tokenizer:

```bash
python -m scripts.train \
    --dataset=tinystories \
    --depth=6 \
    --num-iterations=10000 \
    --batch-size=32 \
    --run=hebbi_tinystories
```

Expected: model generates coherent short stories after ~5000 steps.

### Stage 2: Pretrain on ClimbMix (Full Scale)

400B-token web+books+code dataset for the full 100M-param model:

```bash
# Single GPU (H100: ~4-8h, A100: ~8-16h)
python -m scripts.train \
    --dataset=climbmix \
    --depth=12 \
    --batch-size=32 \
    --grad-accum=4 \
    --num-iterations=50000 \
    --run=hebbi_climbmix
```

Use `--compile` with PyTorch 2.0+ for faster training.

### Stage 3: SFT on SmolTalk

Fine-tune on conversations with loss masking (only assistant tokens):

```bash
python -m scripts.train_sft \
    --checkpoint=checkpoints/hebbi_final.pt \
    --dataset=smoltalk \
    --num-iterations=3000 \
    --run=hebbi_sft
```

### Stage 4: Memory Gate Training

Train each block's memory gate to integrate Hopfield memory bank retrievals.
Each step: forward a context batch → Hebbian write to banks → train on a separate
target batch → clear banks. The gate learns to open when retrieved memories help:

```bash
python -m scripts.train_memory \
    --checkpoint=checkpoints/hebbi_sft_final.pt \
    --dataset=smoltalk \
    --num-iterations=1000 \
    --run=hebbi_memory
```

Gate values start near 0.047 (sigmoid(-3)) and open during training as the model
learns to use retrieved memories.

### Stage 5: Chat with Two-Speed Online Learning

```bash
python -m scripts.chat \
    --checkpoint=checkpoints/hebbi_memory_final.pt \
    --online-learning
```

With GradMem for forgetting protection:

```bash
python -m scripts.chat \
    --checkpoint=checkpoints/hebbi_memory_final.pt \
    --online-learning \
    --n-mem=16
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--depth` | 12 | Number of local blocks (single dial for model size) |
| `--hopfield-steps` | 3 | Energy attention convergence iterations |
| `--ff-threshold` | 2.0 | Forward-Forward goodness threshold |
| `--corruption-rate` | 0.15 | Fraction of tokens corrupted for negatives |
| `--block-lr` | 1e-3 | Per-block learning rate |
| `--head-lr` | 1e-3 | LM head learning rate |
| `--embed-lr` | 1e-2 | Embedding learning rate |
| `--grad-accum` | 1 | Gradient accumulation steps |

## Checkpoints

Checkpoints are saved to `checkpoints/` and contain:
- `config`: model configuration dict
- `model`: model state dict
- `step`: training step number
- `dataset`: dataset used for training

Resume training:

```bash
python -m scripts.train --dataset=climbmix --depth=12 --resume=checkpoints/hebbi_005000.pt
```

## Two-Speed Online Learning

Chat mode implements two learning speeds, inspired by hippocampal-neocortical memory consolidation:

### Fast Path (Every Turn) — Hebbian Memory Banks

After each response, the model performs a reward-modulated Hebbian write to all blocks'
memory banks. This is instant (~0ms, no gradients):

- **good/g** (reward=+1): Strong LTP write — strengthens the association
- **bad/b** (reward=-1): Anti-Hebbian LTD write — actively suppresses the pattern
- **[Enter]** (reward=0): Mild neutral write (strength=0.3)

The reward acts as a dopamine-like neuromodulatory signal. Each write is a single
outer product per block — biologically plausible and computationally free.

### Slow Path (`/sleep`) — FF Weight Consolidation

The `/sleep` command replays logged episodes through full Forward-Forward training,
baking episodic (Hebbian) memories into permanent model weights:

```
/sleep
```

How it works:
1. Replays all logged episodes (with recency-weighted sampling)
2. For each episode: Hebbian write to banks (restore context) → FF weight update
3. Uses the original reward signal from each episode
4. Clears banks after sleep (fresh start)
5. Configurable: `--sleep-epochs=3` (default)

Episodes persist across sessions in `.episodes.jsonl` files next to the checkpoint,
so `/sleep` can consolidate memories from multiple conversations.

### Episode Logging

Every conversation turn is logged with its reward and timestamp to
`<checkpoint>.episodes.jsonl`. This enables:
- Cross-session memory consolidation via `/sleep`
- Replay with original rewards (not re-evaluated)
- Recency bias: recent episodes are replayed more often

### Save on Exit

If any learning occurred (Hebbian writes or /sleep), the model is saved to
`<checkpoint>.chat.pt` on exit. Resume with:
```bash
python -m scripts.chat --checkpoint=checkpoints/hebbi_memory_final.chat.pt --online-learning
```

## GradMem (Forgetting Protection)

When `--n-mem > 0`, learnable prefix memory tokens are prepended to the input. During chat:

1. Before each response, memory tokens are adapted via gradient descent on the conversation context
2. Only memory tokens change — base model weights stay frozen during adaptation
3. This compresses conversation history into a fixed-size memory prefix

This prevents catastrophic forgetting during online learning.

## Monitoring with Wandb

```bash
pip install wandb
python -m scripts.train --dataset=tinystories --depth=6 --run=my_experiment
```

Tracked metrics:
- `train/lm_loss`: Language modeling loss (smoothed)
- `train/ff_avg`: Average Forward-Forward loss across blocks
- `train/goodness_pos_*`: Per-block positive goodness (should be > threshold)
- `train/goodness_neg_*`: Per-block negative goodness (should be < threshold)
- `val/loss`: Validation cross-entropy loss

## Cloud GPU Setup

### Lambda Cloud / RunPod

```bash
# SSH into GPU instance, then:
git clone https://github.com/Findeton/hebbi.git
cd hebbi
pip install -e .
pip install wandb

# Full training run
python -m scripts.train --dataset=climbmix --depth=12 --compile --run=hebbi_full
```

### Google Colab

```python
!git clone https://github.com/Findeton/hebbi.git
%cd hebbi
!pip install -e .
!python -m scripts.train --dataset=tinystories --depth=6 --num-iterations=5000
```
