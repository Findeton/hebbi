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
    --num-iterations=2000 \
    --run=hebbi_sft
```

### Stage 4: Chat with Online Learning

```bash
python -m scripts.chat \
    --checkpoint=checkpoints/hebbi_sft_final.pt \
    --online-learning
```

With GradMem for forgetting protection:

```bash
python -m scripts.chat \
    --checkpoint=checkpoints/hebbi_sft_final.pt \
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

## Online Learning

During chat, user feedback modulates the Forward-Forward learning signal:

- **good/g**: Response becomes a strongly positive example (reward=+1)
- **bad/b**: Response becomes a negative example — the model learns "this is NOT good text" (reward=-1)
- **[Enter]**: Mild neutral update (reward=0)

This is analogous to dopamine modulation in biological neural circuits.

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
