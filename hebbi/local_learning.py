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


def generate_negatives_predictive(model, input_ids, corruption_rate=0.15, rng=None):
    """
    Predictive coding negatives: replace tokens with the model's own predictions.

    Instead of random token replacement, we use what the model thinks should be
    there. This is much harder to detect — the blocks must learn deep language
    structure, not just spot statistical outliers.

    Biologically: this is predictive coding. The negative is "what I predicted,"
    the positive is "what actually happened." The blocks learn to compute
    prediction errors — exactly what cortical columns are thought to do.

    As the model improves, its predictions become more plausible, so negatives
    get harder automatically. Natural curriculum with no scheduling needed.
    """
    B, T = input_ids.shape
    device = input_ids.device

    # Get model predictions (no gradients, inference only)
    with torch.no_grad():
        logits = model(input_ids)  # (B, T, V)

    # Sample from predictions (shifted right: logits[:, t] predicts token t+1)
    # For position t, we use logits[:, t-1] (what the model predicted for position t)
    # For position 0, we fall back to random (no prediction available)
    probs = F.softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
    predicted = torch.multinomial(
        probs.reshape(-1, probs.size(-1)), num_samples=1
    ).reshape(B, T - 1)  # (B, T-1)

    # Pad position 0 with a random token (no prediction for the first token)
    rng_device = rng.device if rng is not None else device
    first_token = torch.randint(
        0, logits.size(-1), (B, 1), device=rng_device, generator=rng
    ).to(device)
    predicted = torch.cat([first_token, predicted.to(device)], dim=1)  # (B, T)

    # Apply corruption mask (same rate as random negatives)
    mask = torch.rand(B, T, device=rng_device, generator=rng) < corruption_rate
    mask = mask.to(device)
    neg_ids = torch.where(mask, predicted, input_ids)
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


class AdaptiveThreshold:
    """
    Adaptive FF threshold that tracks goodness and stays challenging.

    Fixed threshold saturates: once goodness >> θ, FF loss → 0 and blocks
    stop learning. This tracker keeps θ at a fraction of the running average
    of positive goodness, so the objective stays non-trivial throughout training.

    θ_adaptive = max(θ_base, margin_ratio * EMA(g_pos))

    The base threshold is a floor — the adaptive threshold never drops below it.
    margin_ratio < 1 means "keep the threshold just below where goodness is,"
    so blocks always have to push a little harder.

    Checkpoint-safe: state_dict() / load_state_dict() for save/resume.
    """

    def __init__(self, base_threshold=2.0, margin_ratio=0.5, ema_decay=0.999,
                 warmup_steps=500):
        self.base_threshold = base_threshold
        self.margin_ratio = margin_ratio
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.goodness_ema = 0.0
        self.n_updates = 0

    def update(self, goodness_pos_mean):
        """Update EMA with the current step's mean positive goodness."""
        if self.n_updates == 0:
            self.goodness_ema = goodness_pos_mean
        else:
            self.goodness_ema = (self.ema_decay * self.goodness_ema
                                 + (1 - self.ema_decay) * goodness_pos_mean)
        self.n_updates += 1

    @property
    def threshold(self):
        """Current adaptive threshold with linear warmup from base."""
        if self.n_updates == 0:
            return self.base_threshold
        adaptive = self.margin_ratio * self.goodness_ema
        target = max(self.base_threshold, adaptive)
        # Linear warmup: ramp from base_threshold to target over warmup_steps
        # This prevents a sudden threshold jump when enabling mid-training
        if self.n_updates < self.warmup_steps:
            alpha = self.n_updates / self.warmup_steps
            return self.base_threshold + alpha * (target - self.base_threshold)
        return target

    def state_dict(self):
        return {
            "base_threshold": self.base_threshold,
            "margin_ratio": self.margin_ratio,
            "ema_decay": self.ema_decay,
            "warmup_steps": self.warmup_steps,
            "goodness_ema": self.goodness_ema,
            "n_updates": self.n_updates,
        }

    def load_state_dict(self, state):
        self.base_threshold = state["base_threshold"]
        self.margin_ratio = state["margin_ratio"]
        self.ema_decay = state["ema_decay"]
        self.warmup_steps = state.get("warmup_steps", 500)
        self.goodness_ema = state["goodness_ema"]
        self.n_updates = state["n_updates"]


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

def det_train_step(model, pos_ids, neg_ids, layer_opts, config,
                   adaptive_threshold=None):
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

    Args:
        adaptive_threshold: optional AdaptiveThreshold instance. If provided,
            uses its adaptive threshold instead of config.ff_threshold and
            updates it with this step's mean positive goodness.
    """
    layer_opts.zero_all()
    losses = {}
    threshold = (adaptive_threshold.threshold if adaptive_threshold is not None
                 else config.ff_threshold)

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

    # Update adaptive threshold with this step's mean positive goodness
    if adaptive_threshold is not None:
        g_pos_avg = sum(v for k, v in losses.items()
                        if k.startswith("goodness_pos_")) / len(model.blocks)
        adaptive_threshold.update(g_pos_avg)
        losses["ff_threshold"] = threshold

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
# Backprop baseline training step (for comparison)
# ---------------------------------------------------------------------------

def backprop_train_step(model, pos_ids, layer_opts, config):
    """
    Standard backprop training step — gradients flow through all blocks.

    Uses a single optimizer over all parameters. This exists purely as a
    baseline to measure whether the DET architecture itself can learn,
    independently of whether FF local learning is effective.

    Uses the same LayerOptimizers for convenience but steps all at once.
    """
    layer_opts.zero_all()
    losses = {}

    # Full forward pass WITH gradients through all blocks
    x = norm(model.wte(pos_ids).to(COMPUTE_DTYPE))
    cos_sin = model._get_cos_sin(pos_ids.size(1))

    # Forward through blocks without detaching (gradients flow everywhere)
    x, _ = model.forward_blocks(x, cos_sin, return_goodness=False)

    # LM loss
    logits = model.lm_head(norm(x)).float()
    targets = pos_ids[:, 1:]
    lm_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    lm_loss.backward()

    # Clip and step all optimizers (blocks + head)
    all_params = list(model.parameters())
    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
    for i in range(len(model.blocks)):
        layer_opts.step_block(i)
    layer_opts.step_head()

    losses["lm_loss"] = lm_loss.item()
    return losses


# ---------------------------------------------------------------------------
# Energy-consolidated training step
# ---------------------------------------------------------------------------

def compute_energy_weights(energy_steps, mode="increasing", custom_weights=None):
    """
    Compute normalized per-stage weights for energy-consolidated training.

    Args:
        energy_steps: number of energy iterations
        mode: "increasing" | "uniform" | "decreasing" | "final_only" | "custom"
        custom_weights: list of floats (required when mode="custom")

    Returns:
        list of normalized weights (sum to 1.0)
    """
    if mode == "custom":
        assert custom_weights is not None and len(custom_weights) == energy_steps
        raw = list(custom_weights)
    elif mode == "uniform":
        raw = [1.0] * energy_steps
    elif mode == "increasing":
        # Linear ramp: 1, 2, 3, ... E
        raw = [float(e + 1) for e in range(energy_steps)]
    elif mode == "decreasing":
        # Inverse ramp: E, E-1, ..., 1
        raw = [float(energy_steps - e) for e in range(energy_steps)]
    elif mode == "final_only":
        raw = [0.0] * (energy_steps - 1) + [1.0]
    else:
        raise ValueError(f"Unknown energy_weights mode: {mode}")

    total = sum(raw)
    if total == 0:
        return [0.0] * energy_steps
    return [w / total for w in raw]


def det_train_step_with_energy(model, pos_ids, neg_ids, layer_opts, config,
                                energy_weights):
    """
    Forward-Forward training with recurrent energy dynamics.

    The block stack is traversed energy_steps times. Each pass contributes
    gradients weighted by energy_weights[e] to the SAME block parameters.
    Optimizers step exactly once after the final pass, so block weights stay
    fixed across energy stages within one training iteration — matching
    inference behavior.

    Args:
        model: DET model
        pos_ids: (B, T) positive (real) token ids
        neg_ids: (B, T) negative (corrupted) token ids
        layer_opts: LayerOptimizers instance
        config: DETConfig
        energy_weights: list of E floats — per-stage gradient weights (should
                        be normalized so sum ≈ 1.0)
    """
    layer_opts.zero_all()
    losses = {}
    threshold = config.ff_threshold
    E = len(energy_weights)
    n_layer = len(model.blocks)

    # Initial state: embedding (no grad for embedding in block phase)
    with torch.no_grad():
        x_pos = norm(model.wte(pos_ids).to(COMPUTE_DTYPE))
        x_neg = norm(model.wte(neg_ids).to(COMPUTE_DTYPE))
    cos_sin = model._get_cos_sin(pos_ids.size(1))

    for e in range(E):
        w_e = energy_weights[e]
        x_pos_prev = x_pos
        history_pos = [x_pos]
        history_neg = [x_neg]

        for i, block in enumerate(model.blocks):
            det_hist_pos = [h.detach() for h in history_pos]
            det_hist_neg = [h.detach() for h in history_neg]

            out_pos, g_pos = block(det_hist_pos[-1], cos_sin,
                                   history=det_hist_pos)
            out_neg, g_neg = block(det_hist_neg[-1], cos_sin,
                                   history=det_hist_neg)

            ff_loss = forward_forward_loss(g_pos, g_neg, threshold)

            # Accumulate weighted gradients — no optimizer step yet
            if w_e > 0:
                (w_e * ff_loss).backward()

            history_pos.append(out_pos.detach())
            history_neg.append(out_neg.detach())

            losses[f"ff_{i}_e{e}"] = ff_loss.item()
            losses[f"goodness_pos_{i}_e{e}"] = g_pos.mean().item()
            losses[f"goodness_neg_{i}_e{e}"] = g_neg.mean().item()

        # Feed final output back as next energy stage's input
        x_pos = history_pos[-1]
        x_neg = history_neg[-1]

        # Convergence diagnostics (no extra gradients, just logging)
        with torch.no_grad():
            # Per-stage LM loss: would stopping here give good predictions?
            stage_logits = model.lm_head(norm(x_pos)).float()
            stage_lm = F.cross_entropy(
                stage_logits[:, :-1].reshape(-1, config.vocab_size),
                pos_ids[:, 1:].reshape(-1),
            )
            losses[f"lm_loss_e{e}"] = stage_lm.item()

            # Representation delta (how much changed this pass)
            if e > 0:
                delta = (x_pos - x_pos_prev).square().mean().item()
                losses[f"energy_delta_e{e}"] = delta

    # --- Single optimizer step per block, AFTER all energy stages ---
    for i, block in enumerate(model.blocks):
        torch.nn.utils.clip_grad_norm_(block.parameters(), max_norm=1.0)
        layer_opts.step_block(i)
        layer_opts.block_optimizers[i].zero_grad(set_to_none=True)

    # --- LM head: trains only on the FINAL energy stage's output ---
    logits = model.lm_head(norm(x_pos)).float()
    targets = pos_ids[:, 1:]
    lm_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    emb_for_train = norm(model.wte(pos_ids).to(COMPUTE_DTYPE))
    emb_logits = model.lm_head(emb_for_train).float()
    emb_loss = F.cross_entropy(
        emb_logits[:, :-1].reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    total_head_loss = lm_loss + 0.1 * emb_loss
    total_head_loss.backward()
    head_params = []
    for group in layer_opts.head_optimizer.param_groups:
        head_params.extend(group["params"])
    torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
    layer_opts.step_head()

    losses["lm_loss"] = lm_loss.item()
    losses["emb_loss"] = emb_loss.item()

    # Aggregate per-block across energy stages for convenience
    for i in range(n_layer):
        losses[f"ff_{i}"] = sum(
            losses[f"ff_{i}_e{e}"] for e in range(E)
        ) / E
        losses[f"goodness_pos_{i}"] = losses[f"goodness_pos_{i}_e{E-1}"]
        losses[f"goodness_neg_{i}"] = losses[f"goodness_neg_{i}_e{E-1}"]

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
