"""
Decentralized Energy Transformer (DET)

A biologically-inspired language model that replaces backpropagation with local learning.

Key features:
- Energy-based attention via Modern Hopfield Networks (iterative convergence)
- Attention Residuals for dynamic cross-block routing (no fixed residual connections)
- Forward-Forward local learning (each block has its own loss, no gradient between blocks)
- Recurrent energy dynamics (multiple "thinking" iterations)
- GradMem test-time memory adaptation
- ReLU^2 activation, QK norm, rotary embeddings, no learnable norm params
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hebbi.common import COMPUTE_DTYPE


@dataclass
class DETConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304     # GPT-2 BPE (50257) padded to multiple of 64
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 6
    hopfield_beta: float = 1.0  # inverse temperature for energy attention
    hopfield_steps: int = 3     # convergence iterations per attention
    ff_threshold: float = 2.0   # Forward-Forward goodness threshold
    energy_steps: int = 1       # recurrent thinking iterations (Phase 2)
    n_mem: int = 0              # GradMem prefix tokens, 0 = disabled (Phase 2)
    corruption_rate: float = 0.15  # fraction of tokens corrupted for negatives

    @classmethod
    def from_depth(cls, depth, aspect_ratio=64, head_dim=128, **kwargs):
        """Single-dial config: depth controls model size (same pattern as nanochat)."""
        n_embd = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
        n_head = n_embd // head_dim
        return cls(n_layer=depth, n_embd=n_embd, n_head=n_head, **kwargs)


def norm(x):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Master weights stay fp32 for optimizer precision."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def apply_rotary_emb(x, cos, sin):
    """Apply rotary positional embeddings. x: (B, H, T, D)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class EnergyAttention(nn.Module):
    """
    Energy-based attention via Modern Hopfield Network.

    Standard softmax attention is a single-step Hopfield retrieval.
    With hopfield_steps > 1, we iterate to find deeper attractors:

        Energy:  E(x) = -logsumexp(beta * x @ K^T / sqrt(d)) + 0.5 * ||x||^2
        Update:  state_{t+1} = softmax(beta * state_t @ K^T / sqrt(d) + mask) @ V

    The iteration refines the query state before retrieval — a form of
    "thinking" at the attention level.
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.beta = config.hopfield_beta
        self.steps = config.hopfield_steps

        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim

        q = self.c_q(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.c_k(x).view(B, T, H, D).transpose(1, 2)
        v = self.c_v(x).view(B, T, H, D).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK norm (stabilizes attention, same as nanochat)
        q, k = norm(q), norm(k)

        # Causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )

        # Hopfield iteration: refine query state before retrieval
        scale = self.beta / (D ** 0.5)
        state = q
        for _ in range(self.steps):
            scores = scale * (state @ k.transpose(-2, -1))
            scores = scores + causal_mask
            attn = F.softmax(scores, dim=-1)
            state = attn @ v

        y = state.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # relu^2
        x = self.c_proj(x)
        return x


class AttentionResidual(nn.Module):
    """
    Attention Residuals: dynamic depth-wise routing across blocks.

    Instead of fixed residual connections (h_l = h_{l-1} + f(h_{l-1})),
    each block dynamically selects which prior block outputs to attend to:

        h_l = sum_{i=0}^{l-1} alpha_{i->l} * history_i

    where alpha_{i->l} = softmax(w_l^T @ history_i) — a learned per-block
    query attending over all preceding block outputs.

    This prevents signal dilution at depth and enables input-dependent routing.
    """
    def __init__(self, config):
        super().__init__()
        # Learned pseudo-query for this block's depth-wise attention
        self.query = nn.Parameter(torch.randn(config.n_embd))

    def forward(self, history):
        """
        Args:
            history: list of (B, T, D) tensors — outputs from all preceding blocks
                     (including the initial embedding as history[0])
        Returns:
            x: (B, T, D) — weighted combination of history
        """
        # Stack history: (n_prev, B, T, D)
        stacked = torch.stack(history, dim=0)
        n_prev, B, T, D = stacked.shape

        # Compute attention scores: query dot each history entry
        # query: (D,) -> (1, 1, 1, D)
        q = self.query.to(stacked.dtype).view(1, 1, 1, D)
        # scores: (n_prev, B, T, 1)
        scores = (stacked * q).sum(dim=-1, keepdim=True)
        # Normalize across history dimension
        alpha = F.softmax(scores, dim=0)  # (n_prev, B, T, 1)

        # Weighted combination
        x = (alpha * stacked).sum(dim=0)  # (B, T, D)
        return x


class HopfieldMemoryBank(nn.Module):
    """
    Persistent associative memory using Hebbian outer-product storage.

    Stores memories as superimposed key-value patterns in a weight matrix,
    following the Modern Hopfield Network framework. Each block has its own
    private bank — other blocks see retrieved memories only indirectly
    through the attention residual pathway.

    Write: W = decay * W + v ⊗ k     (Hebbian — no gradients)
    Read:  retrieved = gate * W @ norm(x) / sqrt(n_writes)

    The gate parameter is learned via FF loss during memory training (stage 4).
    During stages 1-3 the banks are empty, so read() returns zeros and the
    gate receives zero gradients — training is completely unaffected.

    Capacity: ~D patterns before significant interference (D=768 → ~768
    conversation memories per block). Old memories decay exponentially.
    """

    def __init__(self, n_embd, gate_init=-3.0, decay=0.99):
        super().__init__()
        self.n_embd = n_embd
        self.decay = decay
        # Weight matrix — written via Hebbian rule, not by gradient
        self.register_buffer("W", torch.zeros(n_embd, n_embd))
        self.register_buffer("n_writes", torch.tensor(0, dtype=torch.long))
        # Learned gate: sigmoid(-3) ≈ 0.05 → nearly off at init
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def read(self, x):
        """x: (B, T, D) → (B, T, D) retrieved memory."""
        if self.n_writes.item() == 0:
            return torch.zeros_like(x)
        x_norm = norm(x)
        retrieved = F.linear(x_norm, self.W.to(x.dtype))  # (B, T, D)
        # Scale by 1/sqrt(n_writes) to keep magnitude stable as bank fills
        retrieved = retrieved * torch.rsqrt(self.n_writes.float().clamp(min=1))
        return torch.sigmoid(self.gate) * retrieved

    @torch.no_grad()
    def write(self, key, value, reward=0.0):
        """
        Reward-modulated Hebbian write: W = decay * W + strength * outer(v, k).

        Dopamine-like modulation:
          reward=+1 → strength=1.0  (strong LTP — remember this)
          reward= 0 → strength=0.3  (mild write — default)
          reward=-1 → strength=-0.4 (anti-Hebbian LTD — suppress this pattern)

        key:   (B, T, D) or (D,) — what the block was looking at (query cue)
        value: (B, T, D) or (D,) — what the block computed (useful content)
        reward: float in [-1, 1] — modulates write strength
        """
        # Dopamine-modulated strength: maps [-1, 1] → [-0.4, 1.0]
        strength = 0.3 + 0.7 * reward

        if key.dim() == 3:
            k = norm(key.float().mean(dim=(0, 1)))   # (D,)
            v = value.float().mean(dim=(0, 1))        # (D,)
        elif key.dim() == 2:
            k = norm(key.float().mean(dim=0))
            v = value.float().mean(dim=0)
        else:
            k = norm(key.float())
            v = value.float()
        self.W.mul_(self.decay).add_(torch.outer(v, k), alpha=strength)
        self.n_writes.add_(1)

    @torch.no_grad()
    def clear(self):
        """Reset the bank to empty state."""
        self.W.zero_()
        self.n_writes.zero_()


class LocalBlock(nn.Module):
    """
    A semi-autonomous block with energy attention, MLP, Hopfield memory,
    and attention residuals.

    Each block:
    - Selects its input via attention residuals (dynamic routing from all prior blocks)
    - Processes via energy-based attention + MLP
    - Reads from its persistent Hopfield memory bank
    - Reports goodness for Forward-Forward training
    - Learns independently with its own optimizer
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = EnergyAttention(config)
        self.mlp = MLP(config)
        self.memory_bank = HopfieldMemoryBank(config.n_embd)
        # Attention residual: dynamic routing from preceding blocks
        # (not needed for block 0 — it just uses the embedding directly)
        if layer_idx > 0:
            self.attn_resid = AttentionResidual(config)
        else:
            self.attn_resid = None

    def forward(self, x, cos_sin, history=None):
        """
        Args:
            x: (B, T, D) — input (used directly for block 0, or as fallback)
            cos_sin: rotary embedding tuple
            history: list of prior block outputs (for attention residuals)

        Returns:
            output: (B, T, D)
            goodness: (B, T) — mean squared activation norm for FF training
        """
        # Attention residual routing (blocks 1+)
        if self.attn_resid is not None and history is not None and len(history) > 0:
            x = self.attn_resid(history)

        # Attention + MLP + Memory with residual connections (within-block only)
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.memory_bank.read(x)
        x = x + self.mlp(norm(x))

        # Goodness: mean squared activation norm per position
        # (computed on raw x so it carries real magnitude signal for FF)
        goodness = x.square().mean(dim=-1)  # (B, T)

        # CRITICAL for Forward-Forward stability: normalize the output that
        # gets passed on to the next block, so downstream blocks only see the
        # *direction* of this block's activation, not its magnitude. Without
        # this, FF optimizers discover the degenerate solution of inflating
        # activations to raise goodness, and training diverges. This is
        # Hinton's original FF normalization trick.
        x_out = norm(x)
        return x_out, goodness


class DET(nn.Module):
    """
    Decentralized Energy Transformer.

    A language model where each block learns independently via Forward-Forward,
    blocks communicate via attention residuals, and attention uses energy-based
    Hopfield dynamics.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [LocalBlock(config, i) for i in range(config.n_layer)]
        )
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        # GradMem: learnable prefix memory tokens (Phase 2)
        if config.n_mem > 0:
            self.memory = nn.Parameter(
                torch.randn(1, config.n_mem, config.n_embd) * 0.02
            )

        # Precompute rotary embeddings
        self.rotary_seq_len = config.sequence_len * 4
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().to(COMPUTE_DTYPE)
        sin = freqs.sin().to(COMPUTE_DTYPE)
        # Shape: (1, 1, T, D/2) for broadcasting with (B, H, T, D/2)
        return cos[None, None, :, :], sin[None, None, :, :]

    @torch.no_grad()
    def init_weights(self):
        # Embedding
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.8)
        # LM head
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.blocks:
            for name in ["c_q", "c_k", "c_v"]:
                torch.nn.init.uniform_(getattr(block.attn, name).weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            # Attention residual query
            if block.attn_resid is not None:
                torch.nn.init.normal_(block.attn_resid.query, std=0.02)
        # Rotary embeddings (recompute on actual device)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Embedding dtype
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)

    def _get_cos_sin(self, T, offset=0):
        """Get rotary embeddings for sequence length T starting at offset."""
        return self.cos[:, :, offset:offset + T], self.sin[:, :, offset:offset + T]

    def forward_blocks(self, x, cos_sin, return_goodness=False):
        """
        Forward pass through all blocks with attention residuals.

        For FF training: each block receives a DETACHED history, so
        gradients don't flow between blocks.

        Returns:
            x: (B, T, D) final output
            goodness_list: list of (B, T) goodness per block
        """
        goodness_list = []
        history = [x]  # history[0] = embedding output

        for i, block in enumerate(self.blocks):
            # For FF training: detach history so gradients stay local
            if return_goodness:
                block_history = [h.detach() for h in history]
                block_input = block_history[-1]
            else:
                block_history = history
                block_input = history[-1]

            output, goodness = block(block_input, cos_sin, history=block_history)

            if return_goodness:
                goodness_list.append(goodness)

            history.append(output if not return_goodness else output.detach())

        return history[-1], goodness_list

    def forward_and_collect(self, idx):
        """
        Forward pass that returns per-block (input, output) pairs.
        Used by memory training to populate banks with Hebbian writes.
        """
        B, T = idx.size()
        cos_sin = self._get_cos_sin(T)
        x = norm(self.wte(idx).to(COMPUTE_DTYPE))

        block_ios = []  # list of (input, output) per block
        history = [x]
        for block in self.blocks:
            block_input = history[-1]
            output, _ = block(block_input, cos_sin, history=history)
            block_ios.append((block_input.detach(), output.detach()))
            history.append(output.detach())
        return block_ios

    def write_to_banks(self, block_ios, reward=0.0):
        """Write per-block (input, output) pairs into memory banks."""
        for i, block in enumerate(self.blocks):
            key, value = block_ios[i]
            block.memory_bank.write(key, value, reward=reward)

    def clear_banks(self):
        """Clear all memory banks."""
        for block in self.blocks:
            block.memory_bank.clear()

    def forward(self, idx, targets=None):
        """Full forward pass for inference/eval."""
        B, T = idx.size()
        cos_sin = self._get_cos_sin(T)

        x = self.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # GradMem: prepend memory tokens
        if self.config.n_mem > 0 and hasattr(self, "memory"):
            mem = self.memory.expand(B, -1, -1).to(x.dtype)
            x = torch.cat([mem, x], dim=1)
            cos_sin = self._get_cos_sin(x.size(1))

        # Recurrent energy dynamics: multiple passes (Phase 2)
        for energy_iter in range(self.config.energy_steps):
            x, _ = self.forward_blocks(x, cos_sin, return_goodness=False)

        # Remove memory prefix
        if self.config.n_mem > 0 and hasattr(self, "memory"):
            x = x[:, self.config.n_mem :, :]

        x = norm(x)
        logits = self.lm_head(x).float()

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="mean",
            )
            return loss
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Naive autoregressive generation (same pattern as nanochat)."""
        assert isinstance(tokens, list)
        device = self.wte.weight.device
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            ids_cond = ids[:, -self.config.sequence_len :]
            logits = self.forward(ids_cond)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            yield next_id.item()
