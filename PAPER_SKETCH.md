# Sleep Is All You Need

## Learning Without Backpropagation via Forward-Forward, Hopfield Memory, and Hebbian Consolidation

**Status:** Sketch. Architecture implemented, experiments not yet run. Numbers, plots, and ablations are placeholders. This document captures the claims we expect to test and the story we expect to tell.

---

## Abstract

We present Hebbi, a language model trained entirely with local learning rules — no backpropagation between layers — that augments each block with a persistent Hopfield memory bank and consolidates episodic memories into weights via a biologically-plausible "sleep" procedure. Four mechanisms combine to produce a model that can (i) extend its effective context through associative recall, (ii) think longer via recurrent energy dynamics without inflating the context window, (iii) adapt online from conversational feedback at zero gradient cost, and (iv) consolidate those adaptations into permanent weights during an offline sleep phase that requires no stored text — only the Hopfield state. We argue that this combination partially addresses catastrophic forgetting, moves beyond the static-after-pretraining paradigm, and offers a path toward models that learn continuously across sessions. The central conjecture: **sleep is the missing primitive** for continual learning with local rules.

---

## 1. Introduction

Modern LLMs are frozen after pretraining. Context windows grow, but the weights stay fixed. Online learning is hamstrung by (a) backpropagation's global synchrony requirements, (b) catastrophic forgetting when gradients overwrite prior knowledge, and (c) the lack of a principled offline consolidation phase.

Biology solves these problems differently. Cortical learning is local (Hebbian, STDP), the hippocampus provides a fast associative store that complements slower cortical weights, and sleep replays hippocampal patterns to consolidate them into neocortex. We take these three ideas seriously and build a language model around them.

Our thesis, stated strongly: **given local learning (Forward-Forward), fast associative memory (Hopfield banks), and an offline consolidation phase (sleep), you recover most of what backpropagation-based LLMs do — plus continual learning, which they cannot do.**

### Contributions

1. **Hebbi architecture** — a transformer-flavored language model trained end-to-end with Forward-Forward local learning, with per-block Hopfield memory banks and learned gates controlling their influence.
2. **Reward-modulated Hebbian writes** — a zero-gradient online learning rule driven by a dopamine-like scalar feedback signal.
3. **Hebbian sleep** — an offline consolidation procedure that replays from Hopfield state alone (no stored text) via dream generation, and distills bank contributions into block weights using a student/teacher FF loss.
4. **Three-timescale memory** — a unified framework combining fast (Hopfield), medium (GradMem prefix tokens), and slow (FF weight updates) adaptation, mapped to biological analogs.
5. **System-2 dynamics without longer context** — recurrent energy iterations that deepen reasoning by refining the attention attractor, not by appending tokens.

---

## 2. Background & Motivation

### 2.1 Forward-Forward Learning
Hinton's Forward-Forward algorithm replaces global backprop with per-layer goodness objectives: positive data should produce high activity norms, negative (corrupted) data should produce low ones. Gradients never cross layer boundaries. This removes the two main obstacles to continual and on-device learning: global synchronization and weight transport.

Previous work has shown FF is competitive on small classification tasks but struggles to scale to language modeling. We report what we think is needed to make it work at scale: output normalization, per-block gradient clipping, and conservative LRs (§4.3).

### 2.2 Modern Hopfield Networks
Modern continuous Hopfield networks are mathematically equivalent to transformer attention at their fixed point. Iterating the Hopfield update beyond one step corresponds to deepening the energy descent — the network "thinks harder" at each layer without any change to the input sequence.

### 2.3 The Catastrophic Forgetting Problem
Continual learning with gradient-based methods degrades earlier knowledge. Existing remedies — rehearsal, EWC, adapters — each sacrifice something (memory, plasticity, parameter efficiency). We sidestep the tradeoff by having learning happen in two places: a fast associative store that can be rewritten freely, and slow weights that are only touched during a controlled sleep phase.

---

## 3. Hebbi: Architecture

We describe the model bottom-up: a block's five subcomponents, then how blocks compose, then the three-timescale memory system built on top.

### 3.1 Notation

- `B` batch, `T` sequence length, `D = n_embd`, `H = n_head`, `d = D/H` head dim.
- `L = n_layer` blocks indexed `l = 0..L-1`.
- `norm(x)` is parameter-free RMSNorm: `x · rsqrt(mean(x²) + ε)`. No learnable scale or bias anywhere in the model — this matters for FF stability (§3.7).
- `σ(·)` is sigmoid. `⊗` is outer product.

Concrete numbers for the 100M-param configuration we target: `D = 768, H = 6, d = 128, L = 12, seq_len = 1024, vocab = 50304` (GPT-2 BPE padded to a multiple of 64).

### 3.2 Embedding and LM Head

Input tokens `idx ∈ ℤ^{B×T}` are embedded and immediately normalized:

```
x₀ = norm(wte(idx))                 ∈ ℝ^{B×T×D}
```

The LM head is a plain linear `lm_head : ℝ^D → ℝ^V` applied to `norm(x_L)`, not weight-tied to the embedding. Weight-tying would create a gradient coupling between head and embedding that we prefer to keep explicit and independent, since they are trained by separate optimizers with separate learning rates (`embed_lr = 10 × head_lr`).

Initialization: `wte ~ N(0, 0.8²)` (deliberately large — we want the embedding to dominate early training before blocks wake up), `lm_head ~ N(0, 0.001²)` (small — the head starts near-uniform so FF signal isn't swamped by random classification noise).

### 3.3 A Block, End to End

A block is *not* a classical PreNorm transformer layer. It is a small self-contained learner with five distinct steps. All of them happen inside one block before control passes to the next:

```
Input:  block_input ∈ ℝ^{B×T×D}, history = [x₀, x₁, ..., x_{l-1}]

# Step 1: Attention-residual routing (blocks ≥ 1 only)
if l > 0:
    x = AttentionResidual_l(history)        # learned softmax over prior block outputs
else:
    x = block_input                         # block 0 uses the embedding directly

# Step 2: Energy attention (multi-step Hopfield)
x = x + EnergyAttention_l(norm(x))          # QK-normed, RoPE, causal, β, K iterations

# Step 3: Hopfield memory read
x = x + MemoryBank_l.read(x)                # σ(gate_l) · (W_l · norm(x)) / √n_writes

# Step 4: MLP (ReLU² feedforward)
x = x + MLP_l(norm(x))                      # 4× expansion, ReLU², no bias

# Step 5: Goodness and output normalization
goodness_l = mean(x²) across D              # FF objective target
x_out = norm(x)                             # magnitude-stripped, direction preserved

Return: (x_out, goodness_l)
```

Every `+` in the list above is a *within-block* residual. Between blocks there is no classical residual at all — downstream blocks reach prior-block state through the attention-residual mechanism (§3.3.1), not by summing.

#### 3.3.1 Attention residuals (cross-block routing)

Each block `l ≥ 1` holds a learned pseudo-query `q_l ∈ ℝ^D`. Given the history tensor `H_l = stack(x₀, x₁, ..., x_{l-1}) ∈ ℝ^{l×B×T×D}`, the block computes:

```
scores_i  = ⟨H_l[i], q_l⟩                   ∈ ℝ^{B×T}           for i ∈ [0, l)
α_{l,i}   = softmax_i(scores_i)             ∈ ℝ^{B×T}
x         = Σ_i α_{l,i} · H_l[i]
```

`α_{l,i}` is computed per-position and per-batch-element, so the routing is input-dependent — a token that needs low-level signal draws from early blocks, a token that needs high-level abstraction draws from late blocks. This replaces the fixed `h_l = h_{l-1} + f(...)` recurrence with learned depth-wise attention. `q_l` is initialized `N(0, 0.02²)`.

Motivation: PreNorm transformers at depth suffer from signal dilution — early-block contributions decay exponentially as they pass through later residuals. Attention residuals make every block's output directly reachable by every later block with unit-scale weight if needed.

#### 3.3.2 Energy attention (multi-step Hopfield)

Standard attention is a one-step Hopfield retrieval; iterating deepens the energy descent. For input `x` we compute Q/K/V with RoPE applied and QK-normalized:

```
q, k, v = split_heads(x W_Q), split_heads(x W_K), split_heads(x W_V)
q, k    = RoPE(q), RoPE(k)
q, k    = norm(q), norm(k)                  # per-head RMSNorm for attention stability
```

Let `scale = β / √d` with `β = hopfield_beta` (default 1.0). Starting from `state_0 = q`, iterate for `K = hopfield_steps` steps (default 3):

```
for t = 0..K-1:
    scores_{t+1} = scale · state_t · k^T + causal_mask
    attn_{t+1}   = softmax(scores_{t+1})
    state_{t+1}  = attn_{t+1} · v
output = state_K · W_O
```

At `K = 1` this is standard softmax attention (no free parameters introduced). At `K > 1` the query is being refined against the same keys iteratively — each step pulls the state toward a deeper Hopfield attractor. The network's "thinking depth" at a single layer becomes a runtime knob (no weights or tokens added). We expect `K = 3` to be cheap enough to leave on by default; varying `K` at inference is one of our headline experiments (§6.3).

`W_O` (here `c_proj`) is **zero-initialized** so the block starts the attention residual close to identity — this is critical for FF stability at init.

#### 3.3.3 MLP

Four-times expansion with ReLU² activation and no biases:

```
MLP(x) = W_down · (ReLU(W_up · x))²
```

`W_up` is initialized uniform on `[-s·0.4, s·0.4]` with `s = √3 · D^{-1/2}`. `W_down` is zero-initialized, again so the block starts near identity.

#### 3.3.4 Hopfield memory read

Each block owns a persistent associative matrix `W_l ∈ ℝ^{D×D}` and a learned scalar gate `gate_l ∈ ℝ`:

```
if n_writes_l == 0:
    return 0                                # bank is empty → exact zero output
retrieved = (W_l · norm(x)) · (n_writes_l)^{-1/2}
return σ(gate_l) · retrieved
```

Three important properties:

1. **Query-side normalization.** Retrieval uses `norm(x)` as the query, so bank reads are scale-invariant in `x`. This prevents a pathology where a block inflates its activations to amplify its own bank contribution.
2. **Writes-count scaling.** Dividing by `√n_writes_l` keeps retrieval magnitude stable as the bank fills — without this, the retrieved term's magnitude would grow linearly with the number of episodes written, eventually dominating the residual stream.
3. **Closed-at-init gate.** `gate_l` starts at `-3`, so `σ(gate_l) ≈ 0.047`. During pretraining, banks are empty anyway (`read` returns zero, giving zero gradient to `gate_l`), so the gate stays dormant until the dedicated memory-training stage (§4.1, stage 4) explicitly opens it.

During pretraining and SFT, `read()` returns exact zero and the memory read is a no-op — so stages 1-3 are compatible with any checkpoint that predates the memory bank. This is how we're able to add banks to a model mid-pipeline without invalidating earlier weights.

#### 3.3.5 Hopfield memory write (Hebbian, no gradients)

Memory writes happen outside the forward pass, triggered by the online-learning loop or by sleep. Given a key `k ∈ ℝ^D` and value `v ∈ ℝ^D` (both obtained by averaging a block's pre- and post-step-2 activations over batch and time), and a scalar reward `r ∈ [-1, 1]`:

```
strength = 0.3 + 0.7 · r                    # r=+1 → 1.0, r=0 → 0.3, r=-1 → -0.4
k        = norm(k)                          # unit-norm key
W_l      ← decay · W_l + strength · (v ⊗ k) # decay = 0.99
n_writes_l += 1
```

This is a classical Hebbian outer-product update, one per block per write, with three biologically-motivated choices:

- **Dopamine-like modulation.** `r = -1` produces a *negative* strength (anti-Hebbian / LTD), actively suppressing the stored pattern rather than just failing to strengthen it.
- **Decay.** `decay = 0.99` gives an effective half-life of ~70 writes, preventing unbounded accumulation while still letting recent episodes dominate.
- **Key normalization, value not.** The key lives on the unit sphere (so retrieval is cosine-similarity-like), the value preserves magnitude (so content can carry weight).

No gradient ever touches `W_l` — it is a `register_buffer`, not a `Parameter`. The only trainable component of the memory bank is the scalar `gate_l`.

#### 3.3.6 Goodness and output normalization (FF glue)

After the MLP residual, the block computes its FF training signal:

```
goodness_l = mean(x²) along D               # shape: (B, T)
```

Positive (real) data should push `goodness_l > θ`, negative (corrupted) data should push it below. But — and this is the stability crux — `goodness_l` is unbounded above, so SGD on `softplus(-(goodness_pos - θ))` can be trivially satisfied by scaling `x` to infinity.

The fix is one line: **the output passed to downstream blocks is `norm(x)`, not `x` itself.** Downstream computation (attention residuals, energy attention, MLP) sees only the *direction* of the block's activation, never its magnitude. Inflating `x` to raise goodness no longer helps, because any downstream use strips the magnitude first. This is Hinton's original FF trick; omitting it causes reliably divergent training around step 150 (§4.3).

### 3.4 Block-by-block forward pass

Given token ids, the full forward pass is:

```
x₀ = norm(wte(idx))
history = [x₀]
for l in 0..L-1:
    if train_mode:
        block_history = [h.detach() for h in history]   # gradient isolation
    else:
        block_history = history
    x_l, goodness_l = block_l(block_history[-1], cos_sin, history=block_history)
    history.append(x_l.detach() if train_mode else x_l)
logits = lm_head(norm(history[-1]))
```

Two training modes, one forward function:

- **Inference / eval mode** — history accumulates without detach. The model is a plain (if unusual) language model.
- **FF training mode** — each block receives a *detached* copy of the history. A block's gradient never reaches its inputs, which means it never crosses a block boundary. Each block is trained by its own local FF objective, its own optimizer, its own `.backward()` call. There is no global loss, no end-to-end backprop, and deleting a block doesn't require re-training the others.

This is not an implementation trick. It is the architectural commitment: blocks are autonomous learners wired together by detached activation routing. Everything else in the system — three-timescale memory, Hebbian sleep, online learning — is built on this commitment.

### 3.5 Recurrent energy dynamics (inference-time thinking)

At inference, the `forward_blocks` call can be repeated `E = energy_steps` times with the output of one pass feeding the embedding of the next:

```
for e in 0..E-1:
    x, _ = forward_blocks(x, cos_sin)
```

Each outer pass is one more round of energy minimization across the full block stack. This is orthogonal to `hopfield_steps` (which deepens attention *within* a block): `energy_steps` deepens the entire network's energy descent, while `hopfield_steps` deepens each attention head's attractor individually. Both are compute knobs that extend thinking depth *without lengthening the token sequence*.

Concrete implication: to "think harder" about a prompt, you can crank `energy_steps` or `hopfield_steps` at inference with no change to context, no extra tokens generated, and no additional parameters. Chain-of-thought is a complement, not a substitute — this operates at a different level.

**Training support.** While the architecture *permits* iteration without training for it, we explicitly train the model to benefit from it via an energy consolidation stage (§4.4). This stage runs the block stack `E` times per training step with gradient accumulation across passes — weights stay fixed within one iteration (matching inference), and later passes get higher gradient weight so the model is trained to improve through iteration, not just maintain consistency.

### 3.6 Single-dial scaling

We expose one knob for model size, same pattern as nanochat:

```
n_embd = round_up(depth · 64, 128)          # aspect ratio 64, rounded to head_dim
n_head = n_embd / 128
n_layer = depth
```

So `--depth=3` → 3 layers, `D = 192`; `--depth=6` → 6 layers, `D = 384`; `--depth=12` → 12 layers, `D = 768`. All other hyperparameters are held fixed. This gives us a clean scaling axis for §6.1.

### 3.7 Initialization summary

| Component | Init |
|-----------|------|
| `wte` | `N(0, 0.8²)` — large, dominates early training |
| `lm_head` | `N(0, 0.001²)` — small, near-uniform output at init |
| Attention `W_Q, W_K, W_V` | `U(-s, s)` with `s = √3 · D^{-1/2}` |
| Attention `W_O` (`c_proj`) | **zero** — attention residual ≈ 0 at init |
| MLP `W_up` (`c_fc`) | `U(-0.4s, 0.4s)` |
| MLP `W_down` (`c_proj`) | **zero** — MLP residual ≈ 0 at init |
| Attention-residual query `q_l` | `N(0, 0.02²)` |
| Memory bank `W_l` | zero buffer (not a parameter) |
| Memory bank `gate_l` | scalar `-3` (σ ≈ 0.05, nearly off) |

Zero-init on the "output" projections of both attention and MLP means the first forward pass through a fresh block is almost exactly identity, and the attention-residual mechanism at init puts most weight on `history[0]` (the embedding) — so a freshly initialized L-layer model behaves like a trivial unigram model whose loss FF training can then drive down without fighting a random initialization.

### 3.8 Three memory timescales

With §3.3–§3.7 in place we now have enough machinery to describe the memory system cleanly:

| Timescale | Where it lives | Update rule | Cost / step | Persistence | Biological analog |
|-----------|----------------|-------------|-------------|-------------|-------------------|
| **Fast** | `W_l` bank (one per block) | Hebbian outer product (§3.3.5) | 1 outer product, no gradients | Until sleep + decay | Hippocampal episodic |
| **Medium** | GradMem prefix tokens `M ∈ ℝ^{n_mem × D}` | Gradient descent on `M` only, blocks frozen | K backward passes through frozen blocks | Per-turn | Working memory |
| **Slow** | Block weights (`W_Q, W_K, W_V, W_O, W_up, W_down, q_l, gate_l`) | Forward-Forward during pretraining, SFT, or sleep (§5) | Full FF training step | Permanent (checkpointed) | Neocortical |

These are not three separate systems bolted together — they are the same physical computation read at three time resolutions. A write to `W_l` immediately biases the next read from `W_l`, which shows up in the block's activations, which in turn affects the FF goodness of the next training step that runs on that block. GradMem sits in the middle: it changes nothing in the weights or the banks but still shifts the block's effective input. The claim is that this stratification is what makes continual learning tractable — fast writes absorb episodes, slow writes absorb patterns, and medium writes provide per-turn context without touching either.

---

## 4. Training

### 4.1 Pipeline
Five sequential stages, each resumable:
1. **TinyStories pretrain** — validates FF at scale with BPE tokenizer.
2. **ClimbMix pretrain** — 400B-token web/books/code.
3. **SmolTalk SFT** — conversations with assistant-token loss masking.
4. **Energy consolidation** — trains the model to use recurrent energy dynamics (`energy_steps=3`). See §4.4.
5. **Memory gate training** — opens the per-block memory gates by running context/target batch pairs through Hebbian write → FF step → bank clear.

Each stage is independently checkpointed. The pipeline bootstraps each stage from the previous stage's weights with the step counter reset, so LR schedules start clean.

### 4.2 Hyperparameters
Single-dial `--depth` config. Per-block AdamW optimizers (FF gradient structure makes Muon-style optimizers inapplicable). `block_lr = head_lr = 3e-4`, `embed_lr = 3e-3`, gradient clipping `max_norm=1.0` per block, warmup 1-3% of steps.

### 4.3 What We Learned About Scaling FF
Unnormalized FF diverges. `goodness = mean(x²)` is unbounded, and optimizers happily inflate activations. The fix — normalizing block outputs and clipping gradients per block — is stated in the Hinton paper but, we think, under-emphasized. Without it we saw predictable explosions around step 100-180. With it, FF is stable at 100M params across 50k steps.

### 4.4 Energy Consolidation Training

Stages 1-3 train with `energy_steps=1` — a single pass through the block stack per training step. This means the model has never been asked to process its own output as its own input. Stage 4 teaches this explicitly.

**Design.** Each training step runs the full block stack `E` times (default 3). Each pass `e ∈ {0, ..., E-1}` computes per-block FF losses and calls `.backward()` with a per-stage weight `w_e`, accumulating gradients into the same `.grad` buffers. Block parameters are not updated between passes — the optimizer steps once, after all `E` passes finish. This is critical: it means block weights are invariant across passes within one training iteration, exactly matching inference behavior where weights are fixed.

**Per-stage weights.** The default is *increasing* (normalized): `w_e = (e+1) / Σ(e+1)`. For `E=3`, this gives `[1/6, 2/6, 3/6] = [0.17, 0.33, 0.50]` — later passes get more gradient weight. The rationale: the model's first-pass behavior is already strong from stages 1-3. What the consolidation stage needs to teach is "don't degrade — in fact, improve — when iterating on your own output." Weighting later passes higher directly rewards that behavior. Normalization ensures the effective learning rate is comparable to earlier stages.

Other weight modes are available (uniform, decreasing, final-only, custom) via a single hyperparameter. The space can be compressed to one scalar `λ ∈ [0,1]` with `w_e = λ^{E-1-e}`, covering the useful range from final-only (`λ=0`) through uniform (`λ=1`).

**What to measure.** The key diagnostic is whether FF loss at energy step `e > 0` is lower than at `e = 0`. If it is, the model is genuinely improving its representations through iteration. If not, the later passes are at best neutral. We log per-stage FF loss and per-stage goodness to track this during training.

**LM head.** Only the final pass's output is shown to the LM head. Exposing intermediate-pass representations would confuse the head with partially-refined states.

**Cost.** An `energy_steps=3` training step takes roughly 3× the compute of a standard step (3 forward+backward passes per block instead of 1). This is acceptable because energy consolidation is short (~2k steps) relative to the full pipeline (~75k steps).

---

## 5. Sleep: Consolidation Without Stored Text

This is the section we are most excited about.

### 5.1 The Problem
Online learning with Hebbian writes is free but volatile — the bank decays, the gate is fixed, and nothing in the weights changes. To *actually learn* from conversations the model must eventually transfer bank contents into block weights. Existing replay schemes store raw text. That's not how brains work, and it raises privacy questions besides.

### 5.2 Hebbian Sleep
Our sleep procedure uses **no stored text** — only the current Hopfield state:

1. **Dream generation.** Sample sequences from the model autoregressively with banks active. Because bank retrievals shape the logits, dreams are colored by what the banks have accumulated during conversation. Dreams are filtered for degeneracy (min length, min unique tokens).
2. **Per-block FF + distillation.** Each dream is fed through blocks with two paths:
   - **Teacher**: banks ON (the configuration that produced the dream).
   - **Student**: bank for *this block* temporarily OFF (bypass by zeroing `n_writes`).
   Each block minimizes `FF_loss + λ · MSE(student, teacher.detach())`.
3. **Local gradient isolation preserved.** Gradients never cross block boundaries — teacher outputs are detached, each block runs its own backward pass, history is rebuilt from detached activations.
4. **Bank decay.** Post-sleep, banks are decayed (not cleared) by a factor (default 0.3), preserving conversation continuity while reflecting that consolidation happened.

### 5.3 Why This Works (Conjecturally)
The distillation loss has a natural termination condition: **as the weights absorb the bank's contribution, `student → teacher` and the distillation loss → 0**. At convergence, the bank is redundant with the weights — exactly the desired outcome of consolidation.

Meanwhile the FF loss prevents dreams from being treated as unconditional positives — the corrupted negatives anchor goodness. This is important: without it, sleep would drift toward whatever the model most readily generates, a classic mode-collapse failure.

### 5.4 Relation to Biological Sleep
- Dream generation ≈ hippocampal replay (the "player" is the bank-colored generator).
- Bank→weight distillation ≈ neocortical consolidation.
- Bank decay ≈ hippocampal forgetting after transfer.
- No stored text ≈ no external tape, only internal state.

We do not claim neural plausibility down to spike timing — only that the *information flow* matches the textbook story.

---

## 6. What We Expect to Measure

*(Placeholders. These are the experiments we plan.)*

### 6.1 Scaling
- FF loss and perplexity vs. model depth on TinyStories and ClimbMix.
- Comparison to backprop-trained baselines of matched param count. We expect a gap — the question is how large and whether it closes with depth.

### 6.2 Hopfield Memory as Extended Context
- Compare perplexity on long documents with (a) standard context, (b) extended context, (c) base context + banks populated from a prefix pass. Hypothesis: banks recover a significant fraction of the perplexity improvement of a literal larger context window, at constant sequence length.
- Ablate the per-block learned gates.

### 6.3 System-2 Thinking (Energy Dynamics)
- **With vs without energy consolidation training.** Compare a model at `energy_steps=3` inference with and without stage 4. Hypothesis: energy consolidation is necessary for the iteration to help; without it, the model's representations collapse toward a fixed point or degrade.
- **Scaling energy steps at inference.** Vary `energy_steps` from 1 to 5 on a consolidated model. Measure perplexity and downstream accuracy vs. compute. Hypothesis: monotonic improvement with diminishing returns — a smooth compute-for-capability tradeoff.
- **Per-stage weight ablation.** Compare increasing, uniform, decreasing, and final-only energy weights on the quality of the consolidated model. Hypothesis: increasing weights are best because they directly reward refinement at later passes.
- **`hopfield_steps` × `energy_steps` grid.** These are orthogonal compute knobs; measure their interaction. Hypothesis: they compose — higher `hopfield_steps` helps each energy step find deeper attractors, and multiple energy steps let the network propagate those deeper attractors across blocks.
- Vary `hopfield_steps` and `energy_steps` at inference on reasoning benchmarks. Hypothesis: inference-time iterations trade compute for capability *without* increasing tokens generated — a compute knob orthogonal to chain-of-thought.

### 6.4 Online Learning
- Chat with "good/bad" feedback. Measure perplexity on held-out examples from the conversation's topic before and after.
- Forgetting probe: after learning topic A, chat on topic B, then re-evaluate topic A. Compare with and without GradMem; with and without sleep.

### 6.5 Sleep
- Pre/post-sleep perplexity on dream-source topics.
- Weight-bank redundancy: does distillation loss actually go to zero?
- Catastrophic forgetting resistance: sleep over topic A's banks, then conversations about topic B, then test topic A. Hypothesis: sleep moves learning from volatile banks to stable weights, and the weights are more robust to subsequent bank overwrites than banks alone.
- Repeated sleep: does iterated sleep degrade the model via generative drift, or does the FF+distillation loss hold it stable?

### 6.6 Privacy Property
Since sleep uses no stored text, the only trace of a conversation after sleep lives in (decayed) bank state and updated weights. We expect this to be a feature for on-device deployments.

---

## 7. Catastrophic Forgetting: A Partial Solution

We do not claim to solve catastrophic forgetting. We claim the architecture addresses it along multiple axes and that the combination may be better than any single axis:

| Mechanism | What it protects |
|-----------|------------------|
| FF local learning | Each block learns independently — updates don't cascade globally. |
| Hopfield banks | Fast learning happens here, not in weights, so weights stay stable. |
| GradMem | Per-turn adaptation happens in prefix tokens, not weights. |
| Attention residuals | Learned routing lets blocks specialize without overwriting shared paths. |
| Sleep | Consolidation is controlled and reversible — not every interaction writes weights. |

The failure modes we expect: gate oversaturation, dream collapse during repeated sleep, bank interference when topics are adversarial. We will report these honestly.

---

## 8. Related Work

*(Stubs to expand.)*

- **Forward-Forward / NoProp** — Hinton 2022; Scellier et al on Equilibrium Propagation; local learning methods more broadly.
- **Modern Hopfield Networks** — Ramsauer et al 2021; the attention-as-Hopfield connection.
- **Memory-augmented LMs** — RETRO, Memorizing Transformers, nnPC — compared on: gradient requirements, update speed, and whether the memory is learned.
- **Continual learning** — EWC, synaptic intelligence, experience replay; MERLIN and generative replay as the closest analogs to our sleep procedure.
- **Sleep-inspired consolidation** — CLS (McClelland et al 1995), awake-sleep algorithms, Hinton's Boltzmann machines.
- **Prefix tuning / GradMem** — inference-time adaptation without touching weights.
- **System-2 inference** — chain-of-thought, self-consistency, tree-of-thoughts as *token-level* deliberation vs. ours at the *attention-iteration* level.

---

## 9. Limitations

- **Not yet validated at scale.** We have an architecture and a training pipeline. We do not yet have loss curves on ClimbMix, SFT convergence data, or chat evaluation numbers. Everything in §6 is a placeholder.
- **FF vs backprop gap unknown.** Previous FF results do not guarantee competitive language modeling at 100M+ params.
- **Dream quality bounds sleep.** A model that dreams poorly will consolidate poorly. If generation quality is weak, sleep will amplify the weakness.
- **Bank capacity.** Outer-product writes into a `D×D` matrix have finite capacity. Decay helps but we don't have a principled scaling law.
- **No formal guarantees** on bounded forgetting, stable sleep, or gate convergence.

---

## 10. Discussion: Why "Sleep Is All You Need"

The title is provocative. We don't mean sleep replaces attention (attention is here), or optimization (optimizers are here). We mean something specific: **the missing piece for continual, local-rule learning is an offline consolidation phase.** Without it, Hebbian updates are stuck in volatile banks; with it, they become weights. Every other piece of the system — FF, Hopfield, attention residuals, GradMem — has antecedents in the literature. The combination, and specifically the Hebbian-only sleep procedure, is what we think is new.

If Hebbi works, the implication is that the static-after-pretraining paradigm is a choice, not a constraint. Models can be architectures that keep learning, provided they have a place to put fast writes and a procedure to transfer them.

If Hebbi doesn't work, we still expect to learn something about *which* of the components is load-bearing. We will report either way.

---

## 11. Conclusion

Hebbi is a bet that local learning, fast associative memory, and offline consolidation compose into a continual learner. We have built it, we know how to train it, and we have a concrete path to falsify every claim in this document. The experiments are next.

---

## Appendix A: Implementation Pointers

- Model and banks: [hebbi/model.py](hebbi/model.py)
- FF loss, negatives, per-block optimizers, energy training: [hebbi/local_learning.py](hebbi/local_learning.py)
- Training pipeline: [scripts/run_pipeline.py](scripts/run_pipeline.py)
- Energy consolidation: [scripts/train_energy.py](scripts/train_energy.py)
- Memory-gate training: [scripts/train_memory.py](scripts/train_memory.py)
- Chat with fast-path Hebbian writes and Hebbian sleep: [scripts/chat.py](scripts/chat.py)
- Architecture document: [ARCHITECTURE.md](ARCHITECTURE.md)
- Training guide: [TRAINING.md](TRAINING.md)

## Appendix B: Open Questions

1. Does the distillation loss actually converge to zero during sleep, or does it plateau?
2. What is the minimum dream length and diversity for stable consolidation?
3. Can sleep run unsupervised overnight on a deployed model, or does it need human curation?
4. Does the three-timescale split hold empirically, or do two timescales suffice?
5. How does FF at 1B+ params compare to backprop? Is there a crossover point where the gap closes, or does it widen?
6. Can we formalize "sleep stability" — a condition under which iterated sleep does not degrade the model?
7. Does the model's representation converge to a fixed point under energy iteration, or does it oscillate? If it converges, is the fixed point the same regardless of `energy_steps`?
8. What is the optimal per-stage weight schedule for energy consolidation? Does it depend on model depth or training data?
9. Can energy consolidation be combined with memory gate training in one stage, or do they interfere?
