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

## Phase 2: Continuous Learning & Memory

### Recurrent Energy Dynamics (energy_steps > 1)

Multiple full passes through all blocks before producing output. The energy landscape
settles deeper with more iterations — analogous to "System 2" deliberate thinking.
More iterations at inference = better predictions at the cost of latency.

### GradMem (Test-Time Memory Adaptation)

Learnable prefix memory tokens `M in R^{n_mem x d}` prepended to input.
At test time, freeze model weights, gradient-descend on M to minimize reconstruction
loss on the current context. This enables:
- Adapting to new users/topics without catastrophic forgetting
- Compressing long contexts into fixed-size memory
- Continuous learning from every interaction

### Continual Learning Pipeline

For the chatbot use case:
- Base model trained on text corpus with Forward-Forward
- At inference, GradMem adapts to conversation context
- Periodic consolidation: successful memory states reinforce base weights
- No retraining from scratch — the model evolves continuously

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
det/
├── det/
│   ├── __init__.py
│   ├── common.py           # Device detection, COMPUTE_DTYPE, utilities
│   ├── data.py             # Shakespeare download, CharDataset, data loader
│   ├── model.py            # DETConfig, EnergyAttention, AttentionResidual, LocalBlock, DET
│   └── local_learning.py   # FF loss, negative gen, LayerOptimizers, train step
├── scripts/
│   ├── train.py            # Training loop with per-layer updates
│   └── generate.py         # Inference / text generation
├── ARCHITECTURE.md         # This document
└── pyproject.toml
```
