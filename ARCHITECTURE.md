# Decentralized Energy Transformer (DET)

## Vision

A biologically-inspired language model that replaces backpropagation with local learning rules, 
energy-based dynamics, and continuous adaptation. The ultimate goal is a chatbot that learns 
continuously from every interaction — not a static model frozen after training.

This draws from the research synthesis "Architecting Decentralized Intelligence" which identifies 
7 pillars for next-generation AI:

1. **Attention Residuals** — dynamic depth-wise routing replacing fixed residual connections
2. **Forward-Forward / NoProp** — local layer-wise learning without backpropagation
3. **Temporal Predictive Coding** — continuous temporal processing via prediction error
4. **Energy-Based Attention** — Modern Hopfield Networks / energy minimization
5. **GradMem** — test-time memory adaptation via gradient descent on prefix tokens
6. **Active Inference** — Free Energy Principle for autonomous agency
7. **Recurrent Energy Dynamics** — iterative "System 2" thinking via energy convergence

---

## Phase 1: Core Architecture (Current Implementation)

### Energy-Based Attention (Modern Hopfield Network)

Standard softmax attention is a single-step retrieval from a Hopfield energy landscape.
By iterating multiple steps, we let the energy converge to a deeper attractor:

```
Energy:   E(x) = -logsumexp(beta * x @ K^T / sqrt(d)) + 0.5 * ||x||^2
Update:   state_{t+1} = softmax(beta * state_t @ K^T / sqrt(d) + causal_mask) @ V
```

Standard attention is the 1-step special case. With `hopfield_steps > 1`, the network
performs iterative refinement — a form of "thinking" at the attention level.

### Attention Residuals (Cross-Block Information Routing)

Instead of fixed residual connections (`h_l = h_{l-1} + f(h_{l-1})`), each block
dynamically selects which prior block outputs to attend to:

```
h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
```

where `alpha_{i->l}` is computed via a learned query per block attending over all
preceding block outputs. This:
- Prevents signal dilution at depth (PreNorm dilution problem)
- Enables dynamic, input-dependent routing between blocks
- Mirrors biological selective attention across cortical layers

Implementation: each block maintains a pseudo-query vector. A history buffer stores
all prior block outputs. The block's input is a softmax-weighted combination of the
history, not just the previous block's output.

### Forward-Forward Local Learning

Each block has its own loss function and optimizer. No gradient flows between blocks.

- **Positive data**: real token sequences
- **Negative data**: corrupted sequences (random token replacement at 15% rate)
- **Goodness metric**: mean squared activation norm per position
- **Loss**: `softplus(-(goodness_pos - theta)) + softplus(-(theta - goodness_neg))`
- **Each block uses local `.backward()`** — gradients are computed within that block only
  (input is `.detach()`ed). This is NOT backpropagation through the network.

The embedding and LM head train via a separate local next-character prediction loss.

### Architecture Details

- Character-level (vocab_size ~65 for Shakespeare, extensible to full UTF-8)
- RMSNorm without learnable params
- ReLU^2 activation in MLP
- QK normalization in attention
- Rotary positional embeddings
- Per-block AdamW optimizers (not Muon — FF gradients have different structure)

---

## Phase 2: Continuous Learning & Memory (Current Implementation)

### Hopfield Memory Banks (Hebbian Weight Matrix)

Each block has a persistent memory bank — a learned associative matrix `W ∈ R^{D×D}`
that stores and retrieves episodic memories via reward-modulated Hebbian writes.

```
Write:  W ← decay·W + strength · outer(value, key)
Read:   retrieved = sigmoid(gate) · (W @ norm(x)) / sqrt(n_writes)
```

Key properties:
- **No gradients needed** for writes — pure outer-product Hebbian update (~0ms)
- **Reward-modulated**: `strength = 0.3 + 0.7 * reward` maps reward ∈ [-1, 1] to strength ∈ [-0.4, 1.0]
  - reward=+1 (LTP): strong write, strengthens the association
  - reward=0 (neutral): mild write (strength=0.3)
  - reward=-1 (LTD): anti-Hebbian, actively suppresses the pattern
- **Learnable gate** per block: initialized closed (sigmoid(-3) ≈ 0.047), opens via FF gradients during memory training
- **Decay** (0.99): older memories fade, preventing unbounded growth
- **Scaling** by `1/sqrt(n_writes)`: prevents magnitude explosion as memories accumulate

The gate parameter is the only part that needs gradients — it learns during memory
training (stage 4) whether retrieved memories help each block's FF objective.

### Three Memory Timescales (Biological Analogy)

Hebbi implements three biologically-inspired memory systems:

| Timescale | Mechanism | Biological Analogue | Speed | Persistence |
|-----------|-----------|-------------------|-------|-------------|
| **Fast** | Hopfield Memory Bank (Hebbian writes) | Hippocampus (episodic) | ~0ms, no gradients | Session (cleared on /sleep) |
| **Medium** | GradMem prefix tokens | Working memory | ~50ms, gradient descent | Per-turn adaptation |
| **Slow** | FF weight updates (/sleep consolidation) | Neocortical consolidation (sleep) | ~seconds, full FF training | Permanent (saved to weights) |

### Two-Speed Learning

**Fast path (every conversation turn):**
After each user interaction, the model performs a reward-modulated Hebbian write to
all blocks' memory banks. This is instant (no gradients, no backward pass) — just one
outer product per block. The user's feedback (good/bad/neutral) acts as a dopamine-like
neuromodulatory signal.

**Slow path (`/sleep` command):**
Replays logged episodes through full Forward-Forward training, baking episodic (Hebbian)
memories into permanent (weight) memories. Like hippocampal replay during biological sleep:

1. Each logged episode is first written to memory banks (restoring Hebbian context)
2. Then used for FF weight updates with the original reward signal
3. Recent episodes are replayed more often (recency-weighted sampling)
4. Banks are cleared after sleep (fresh start for new conversations)

Episodes persist across sessions in `.episodes.jsonl` files, so `/sleep` can consolidate
memories from multiple conversations.

### Recurrent Energy Dynamics (energy_steps > 1)

Multiple full passes through all blocks before producing output. The energy landscape
settles deeper with more iterations — analogous to "System 2" deliberate thinking.
More iterations at inference = better predictions at the cost of latency.

### GradMem (Test-Time Memory Adaptation)

Learnable prefix memory tokens `M ∈ R^{n_mem × d}` prepended to input.
At test time, freeze model weights, gradient-descend on M to minimize reconstruction
loss on the current context. This enables:
- Adapting to new users/topics without catastrophic forgetting
- Compressing long contexts into fixed-size memory
- Continuous learning from every interaction

### FF Normalization (Training Stability)

Hinton's original FF paper normalizes block outputs between layers. Without this,
goodness `x.square().mean()` is unbounded and optimizers can inflate activations,
causing divergence (observed as goodness explosion at steps 100-180).

Fix applied in `LocalBlock.forward`:
```python
x = x + self.attn(norm(x), cos_sin)
x = x + self.memory_bank.read(x)
x = x + self.mlp(norm(x))
goodness = x.square().mean(dim=-1)
x_out = norm(x)  # ← normalize output for next block
return x_out, goodness
```

Combined with per-block gradient clipping (`max_norm=1.0`) and conservative learning
rates (`block-lr=3e-4`), this eliminates FF divergence.

---

## Phase 3: Cortical Columns (Future Vision)

The most radical departure: replace the sequential block pipeline entirely with
independent cortical columns operating in parallel, inspired by the Thousand Brains Theory.

### Architecture

```
         +-- column 1 --+
input ---+-- column 2 --+-- lateral voting / consensus --> output
         +-- column 3 --+
         +-- column 4 --+
```

Each column is truly autonomous:
- Receives the same input sequence
- Has its own energy-based attention + MLP
- Maintains its own "belief state" (interpretation of the input)
- Learns independently via purely local rules (Hebbian, no .backward() at all)
- No column needs to wait for another — fully parallel

### Consensus Replaces Depth

Instead of refining through sequential layers, columns **vote**:
- Lateral connections let columns share predictions and resolve disagreement
- Output = consensus representation, not one column's output
- Maps to Active Inference: each column minimizes its own surprise

### Lateral Communication

Options (to be explored):
1. **Cross-column attention**: each column attends to other columns' states
2. **Prediction error passing**: columns send errors to neighbors, Hebbian lateral updates
3. **Energy-based consensus**: global energy over all column states, iterate to convergence

### Fully Gradient-Free Learning

Within each column:
- **Hebbian**: `delta_w = x_pre * x_post` (correlation-based, no gradients)
- **STDP**: spike-timing-dependent plasticity for temporal sequences
- **Perturbation**: add noise to weights, measure effect on local loss, estimate direction

Between columns:
- **Agreement Hebbian**: columns that predict the same next token strengthen lateral connections
- **Prediction error**: disagreement drives lateral weight updates

This eliminates `.backward()` entirely — not just between blocks, but within them.

---

## Design Principles

1. **Minimal and readable** — follow nanochat's style (~500 lines/file, no abstraction bloat)
2. **Explicit over implicit** — no autocast, explicit dtype management
3. **Single-dial config** — `--depth` sets the primary complexity
4. **Character-level first** — self-contained, no tokenizer dependency, fast iteration
5. **Continuous learning native** — the architecture must support learning at inference time
6. **Biologically plausible** — every mechanism should have a neural correlate

---

## File Structure

```
hebbi/
├── hebbi/
│   ├── __init__.py
│   ├── common.py           # Device detection, COMPUTE_DTYPE, DDP utilities
│   ├── data.py             # BPE tokenizer, HF dataset streaming, SFT data loading
│   ├── model.py            # DETConfig, EnergyAttention, HopfieldMemoryBank, LocalBlock, DET
│   └── local_learning.py   # FF loss, negative gen, LayerOptimizers, online learning
├── scripts/
│   ├── train.py            # Pretrain (TinyStories / ClimbMix) with resume
│   ├── train_sft.py        # SFT on conversation data (SmolTalk)
│   ├── train_memory.py     # Memory gate training (stage 4)
│   ├── run_pipeline.py     # Resumable 4-stage training pipeline
│   ├── generate.py         # Text generation from checkpoint
│   └── chat.py             # Interactive chat with two-speed online learning
├── ARCHITECTURE.md         # This document
├── TRAINING.md             # Training guide and pipeline docs
└── pyproject.toml
```
