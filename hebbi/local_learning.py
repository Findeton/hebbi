"""
Forward-Forward local learning for DET.

Each block learns independently via a local goodness metric:
- Positive data (real sequences): goodness should exceed threshold
- Negative data (corrupted sequences): goodness should stay below threshold

No gradient flows between blocks. Each block has its own optimizer.
The embedding and LM head train via a separate local next-token prediction loss.
"""

import torch
import torch.nn.functional as F
from hebbi.common import COMPUTE_DTYPE
from hebbi.model import norm, DET


# ---------------------------------------------------------------------------
# Negative example generation
# ---------------------------------------------------------------------------

def generate_negatives(input_ids, vocab_size, corruption_rate=0.15, rng=None):
    """
    Generate negative examples by corrupting token sequences.

    Replaces corruption_rate fraction of tokens with random tokens.
    This forces each block to learn features that distinguish real
    from corrupted sequences at every level of abstraction.
    """
    B, T = input_ids.shape
    device = input_ids.device
    # Generate on CPU (MPS doesn't support torch.Generator), then move
    rng_device = rng.device if rng is not None else device
    mask = torch.rand(B, T, device=rng_device, generator=rng) < corruption_rate
    random_tokens = torch.randint(0, vocab_size, (B, T), device=rng_device, generator=rng)
    mask = mask.to(device)
    random_tokens = random_tokens.to(device)
    neg_ids = torch.where(mask, random_tokens, input_ids)
    return neg_ids


# ---------------------------------------------------------------------------
# Forward-Forward loss
# ---------------------------------------------------------------------------

def forward_forward_loss(goodness_pos, goodness_neg, threshold):
    """
    Forward-Forward loss for a single block.

    Positive data should have goodness > threshold.
    Negative data should have goodness < threshold.

    Loss = softplus(-(goodness_pos - theta)) + softplus(-(theta - goodness_neg))
    """
    loss_pos = F.softplus(-(goodness_pos - threshold)).mean()
    loss_neg = F.softplus(-(threshold - goodness_neg)).mean()
    return loss_pos + loss_neg


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(step, num_iterations, warmup_steps, warmdown_ratio):
    """Linear warmup, constant, linear warmdown (same pattern as nanochat)."""
    warmdown_iters = round(warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step <= warmdown_start:
        return 1.0
    else:
        progress = (num_iterations - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * 0.05


# ---------------------------------------------------------------------------
# Per-layer optimizer management
# ---------------------------------------------------------------------------

class LayerOptimizers:
    """
    Manages independent AdamW optimizers for each block.

    Each block has its own optimizer that only updates that block's parameters.
    The embedding and LM head get a separate "head" optimizer trained with
    a local next-token prediction loss.
    """

    def __init__(self, model, block_lr=1e-3, head_lr=1e-3, embed_lr=1e-2):
        self.block_optimizers = []
        for block in model.blocks:
            opt = torch.optim.AdamW(
                block.parameters(),
                lr=block_lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
            self.block_optimizers.append(opt)

        # Head optimizer: embedding + LM head (+ memory if present)
        head_params = [
            {"params": list(model.wte.parameters()), "lr": embed_lr},
            {"params": list(model.lm_head.parameters()), "lr": head_lr},
        ]
        if model.config.n_mem > 0 and hasattr(model, "memory"):
            head_params.append({"params": [model.memory], "lr": head_lr})
        self.head_optimizer = torch.optim.AdamW(head_params, weight_decay=0.01)

        # Store initial LRs for scheduling
        self._initial_block_lrs = [
            [g["lr"] for g in opt.param_groups] for opt in self.block_optimizers
        ]
        self._initial_head_lrs = [g["lr"] for g in self.head_optimizer.param_groups]

    def zero_all(self):
        for opt in self.block_optimizers:
            opt.zero_grad(set_to_none=True)
        self.head_optimizer.zero_grad(set_to_none=True)

    def step_block(self, layer_idx):
        self.block_optimizers[layer_idx].step()

    def step_head(self):
        self.head_optimizer.step()

    def update_lr(self, step, num_iterations, warmup_steps, warmdown_ratio):
        """Update learning rates for all optimizers."""
        lrm = get_lr_multiplier(step, num_iterations, warmup_steps, warmdown_ratio)
        for opt, init_lrs in zip(self.block_optimizers, self._initial_block_lrs):
            for g, ilr in zip(opt.param_groups, init_lrs):
                g["lr"] = ilr * lrm
        for g, ilr in zip(self.head_optimizer.param_groups, self._initial_head_lrs):
            g["lr"] = ilr * lrm
        return lrm


# ---------------------------------------------------------------------------
# Core training step
# ---------------------------------------------------------------------------

def det_train_step(model, pos_ids, neg_ids, layer_opts, config):
    """
    One DET training step with local Forward-Forward + LM head.

    Two learning phases, both purely local:

    Phase 1 - Forward-Forward (per-block):
      For each block i:
        - Forward pos and neg through blocks, collecting goodness
        - Inputs to each block are DETACHED from prior blocks
        - Compute FF loss from goodness_pos vs goodness_neg
        - backward() only computes gradients for block i's parameters
        - Step block i's optimizer

    Phase 2 - Language Modeling (embedding + LM head):
      - Take the final block's output (detached from blocks)
      - Cross-entropy loss for next-token prediction
      - Also train embedding with its own local prediction loss
      - Step the head optimizer

    No gradient ever flows from one block to another.
    """
    layer_opts.zero_all()
    losses = {}
    threshold = config.ff_threshold

    # --- Phase 1: Per-block Forward-Forward ---
    # Embed positive and negative (no gradient for embedding in this phase)
    with torch.no_grad():
        emb_pos = norm(model.wte(pos_ids).to(COMPUTE_DTYPE))
        emb_neg = norm(model.wte(neg_ids).to(COMPUTE_DTYPE))

    cos_sin = model._get_cos_sin(pos_ids.size(1))

    # Process through blocks, collecting goodness
    # Each block gets detached inputs from prior blocks
    history_pos = [emb_pos]
    history_neg = [emb_neg]

    for i, block in enumerate(model.blocks):
        # Detach history for local gradient isolation
        det_history_pos = [h.detach() for h in history_pos]
        det_history_neg = [h.detach() for h in history_neg]

        out_pos, g_pos = block(det_history_pos[-1], cos_sin, history=det_history_pos)
        out_neg, g_neg = block(det_history_neg[-1], cos_sin, history=det_history_neg)

        # FF loss for this block
        ff_loss = forward_forward_loss(g_pos, g_neg, threshold)
        ff_loss.backward()
        # Grad clip to prevent runaway updates (FF is very sensitive to spikes)
        torch.nn.utils.clip_grad_norm_(block.parameters(), max_norm=1.0)
        layer_opts.step_block(i)
        layer_opts.block_optimizers[i].zero_grad(set_to_none=True)

        # Store outputs (detached) for attention residuals in later blocks
        history_pos.append(out_pos.detach())
        history_neg.append(out_neg.detach())

        losses[f"ff_{i}"] = ff_loss.item()
        losses[f"goodness_pos_{i}"] = g_pos.mean().item()
        losses[f"goodness_neg_{i}"] = g_neg.mean().item()

    # --- Phase 2: LM head + embedding loss ---
    # LM head: next-token prediction from final block output (detached from blocks)
    x_final = history_pos[-1]  # already detached
    logits = model.lm_head(norm(x_final)).float()
    targets = pos_ids[:, 1:]
    lm_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    # Embedding loss: a local unigram/shallow predictor
    # Trains the embedding by asking: can raw embeddings predict next character?
    emb_for_train = norm(model.wte(pos_ids).to(COMPUTE_DTYPE))
    emb_logits = model.lm_head(emb_for_train).float()
    emb_loss = F.cross_entropy(
        emb_logits[:, :-1].reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    total_head_loss = lm_loss + 0.1 * emb_loss
    total_head_loss.backward()
    # Grad clip the head (embedding + lm_head) optimizer too
    head_params = []
    for group in layer_opts.head_optimizer.param_groups:
        head_params.extend(group["params"])
    torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
    layer_opts.step_head()

    losses["lm_loss"] = lm_loss.item()
    losses["emb_loss"] = emb_loss.item()
    return losses


# ---------------------------------------------------------------------------
# Online learning with reward modulation
# ---------------------------------------------------------------------------

def online_learn_step(model, conversation_ids, layer_opts, config, reward=0.0):
    """
    Online Forward-Forward learning from a single conversation.

    The reward signal modulates learning like a neuromodulatory signal (dopamine):
      +1 (user said "good") -> conversation is strongly positive, stronger update
      -1 (user said "bad")  -> conversation becomes NEGATIVE example (swap pos/neg)
       0 (neutral/default)  -> mild learning, conversation is weakly positive

    Args:
        model: DET model
        conversation_ids: (1, T) tensor of token ids
        layer_opts: LayerOptimizers instance
        config: DETConfig
        reward: float in [-1, 1], modulates learning
    """
    pos_ids = conversation_ids
    neg_ids = generate_negatives(pos_ids, config.vocab_size, config.corruption_rate)

    if reward < 0:
        # User said "bad" — the actual response becomes a negative example
        pos_ids, neg_ids = neg_ids, pos_ids

    # Scale learning rate by reward magnitude
    lr_scale = max(0.1, abs(reward)) if reward != 0 else 0.3

    # Temporarily scale LRs
    saved_lrs = []
    all_opts = layer_opts.block_optimizers + [layer_opts.head_optimizer]
    for opt in all_opts:
        opt_lrs = []
        for g in opt.param_groups:
            opt_lrs.append(g["lr"])
            g["lr"] = g["lr"] * lr_scale
        saved_lrs.append(opt_lrs)

    # Run FF training step
    losses = det_train_step(model, pos_ids, neg_ids, layer_opts, config)

    # Restore LRs
    for opt, opt_lrs in zip(all_opts, saved_lrs):
        for g, lr in zip(opt.param_groups, opt_lrs):
            g["lr"] = lr

    losses["reward"] = reward
    losses["lr_scale"] = lr_scale
    return losses


# ---------------------------------------------------------------------------
# GradMem test-time adaptation (Phase 2)
# ---------------------------------------------------------------------------

def adapt_memory(model, context_ids, n_steps=5, lr=0.01):
    """
    Test-time adaptation: freeze model, optimize memory tokens
    to minimize reconstruction loss on the given context.

    This enables continuous learning without catastrophic forgetting:
    the base model stays frozen, only the memory adapts.
    """
    assert model.config.n_mem > 0, "Model has no memory tokens (n_mem=0)"

    B, T = context_ids.shape

    # Clone memory for this adaptation session
    mem = model.memory.data.clone().requires_grad_(True)
    opt = torch.optim.Adam([mem], lr=lr)

    # Freeze all model parameters so only mem gets updated
    # (no_grad would break the computation graph to mem)
    saved_requires_grad = {}
    for name, p in model.named_parameters():
        saved_requires_grad[name] = p.requires_grad
        p.requires_grad_(False)

    try:
        for _ in range(n_steps):
            opt.zero_grad()
            emb = norm(model.wte(context_ids).to(COMPUTE_DTYPE))
            # Prepend memory — mem keeps gradient flow
            x = torch.cat([mem.expand(B, -1, -1).to(emb.dtype), emb], dim=1)
            cos_sin = model._get_cos_sin(x.size(1))

            # Forward through blocks — activations flow through so
            # gradients can reach mem, but block params are frozen
            for block in model.blocks:
                x, _ = block(x, cos_sin)

            # Reconstruction loss on context portion
            x_ctx = x[:, model.config.n_mem :, :]
            logits = model.lm_head(norm(x_ctx)).float()
            targets = context_ids[:, 1:]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, model.config.vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            opt.step()
    finally:
        # Restore requires_grad state
        for name, p in model.named_parameters():
            p.requires_grad_(saved_requires_grad[name])

    # Install adapted memory
    model.memory.data.copy_(mem.data)
    return loss.item()
