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

### 3.1 Block Structure

Each block is:
```
x ← x + Attention(norm(x))          # Hopfield energy attention, multi-step
x ← x + gate · MemoryBank.read(x)   # learned-gated Hopfield bank read
x ← x + MLP(norm(x))
goodness ← mean(x²)                 # FF objective
x ← norm(x)                         # FF stabilization (§4.3)
```

Cross-block routing uses **attention residuals**: each block attends over the history of all prior block outputs rather than using a fixed residual connection. This prevents PreNorm signal dilution and allows input-dependent depth-wise routing — we expect this to matter more as depth grows.

### 3.2 Hopfield Memory Banks

Each block owns a bank `W ∈ R^{D×D}` updated by reward-modulated outer products:

```
strength ← 0.3 + 0.7 · reward        # reward ∈ [-1, 1]
W ← decay · W + strength · outer(v, k)
retrieved ← σ(gate) · (W @ norm(x)) / √n_writes
```

Key properties:
- **Zero-gradient writes.** One outer product per block. This is the "hippocampus" — fast, cheap, overwritable.
- **Reward-modulated.** `reward=+1` → LTP; `reward=-1` → LTD (anti-Hebbian, actively suppresses the pattern); `reward=0` → mild neutral write.
- **Learned gate.** Initialized closed (σ(-3) ≈ 0.05) and opened by FF gradients during a dedicated memory-training stage, so the model learns *when* to trust retrievals.
- **Decay.** Prevents unbounded accumulation.

### 3.3 Three Memory Timescales

| Timescale | Mechanism | Biology | Cost | Persistence |
|-----------|-----------|---------|------|-------------|
| Fast | Hopfield bank (Hebbian) | Hippocampus | ~0 ms | Until sleep |
| Medium | GradMem prefix tokens | Working memory | ~50 ms | Per-turn |
| Slow | FF weight updates via sleep | Neocortex | Seconds | Permanent |

We conjecture all three are necessary: fast for episodic capture, medium for context-sensitive adaptation, slow for genuine learning.

---

## 4. Training

### 4.1 Pipeline
Four sequential stages, each resumable:
1. **TinyStories pretrain** — validates FF at scale with BPE tokenizer.
2. **ClimbMix pretrain** — 400B-token web/books/code.
3. **SmolTalk SFT** — conversations with assistant-token loss masking.
4. **Memory gate training** — opens the per-block memory gates by running context/target batch pairs through Hebbian write → FF step → bank clear.

Each stage is independently checkpointed. The pipeline bootstraps each stage from the previous stage's weights with the step counter reset, so LR schedules start clean.

### 4.2 Hyperparameters
Single-dial `--depth` config. Per-block AdamW optimizers (FF gradient structure makes Muon-style optimizers inapplicable). `block_lr = head_lr = 3e-4`, `embed_lr = 3e-3`, gradient clipping `max_norm=1.0` per block, warmup 1-3% of steps.

### 4.3 What We Learned About Scaling FF
Unnormalized FF diverges. `goodness = mean(x²)` is unbounded, and optimizers happily inflate activations. The fix — normalizing block outputs and clipping gradients per block — is stated in the Hinton paper but, we think, under-emphasized. Without it we saw predictable explosions around step 100-180. With it, FF is stable at 100M params across 50k steps.

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

### 6.3 System-2 Thinking
- Vary `hopfield_steps` and `energy_steps` at inference. Measure accuracy on reasoning benchmarks as a function of compute. Hypothesis: inference-time iterations trade compute for capability *without* increasing tokens generated — a compute knob orthogonal to chain-of-thought.

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
- FF loss, negatives, per-block optimizers: [hebbi/local_learning.py](hebbi/local_learning.py)
- Training pipeline: [scripts/run_pipeline.py](scripts/run_pipeline.py)
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
