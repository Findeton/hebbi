"""
Data loading for Hebbi.

Supports:
- Character-level Shakespeare (original, for testing)
- BPE tokenizer with chat special tokens
- Streaming data from HuggingFace datasets (TinyStories, ClimbMix, SmolTalk)
- SFT conversation rendering with loss masking
"""

import os
import urllib.request
import torch

# ---------------------------------------------------------------------------
# Character-level Shakespeare (backward compatible)
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharDataset:
    """Simple character-level dataset with encode/decode."""

    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.data = torch.tensor(
            [self.char_to_idx[c] for c in text], dtype=torch.long
        )

    def encode(self, s):
        return [self.char_to_idx[c] for c in s]

    def decode(self, ids):
        return "".join(self.idx_to_char[i] for i in ids)


def get_shakespeare(data_dir="data"):
    """Download and return Shakespeare text."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(filepath):
        print(f"Downloading Shakespeare to {filepath}...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, filepath)
    with open(filepath, "r") as f:
        text = f.read()
    return text


def char_data_loader(data, batch_size, seq_len, device, split="train", split_ratio=0.9):
    """
    Infinite iterator yielding (inputs, targets) of shape (B, T).
    Targets are inputs shifted right by 1 position.
    """
    n = len(data)
    split_idx = int(n * split_ratio)
    if split == "train":
        chunk = data[:split_idx]
    else:
        chunk = data[split_idx:]

    n = len(chunk)
    while True:
        starts = torch.randint(0, n - seq_len - 1, (batch_size,))
        x = torch.stack([chunk[s : s + seq_len] for s in starts]).to(device)
        y = torch.stack([chunk[s + 1 : s + seq_len + 1] for s in starts]).to(device)
        yield x, y


# ---------------------------------------------------------------------------
# BPE Tokenizer with chat special tokens
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
]


class HebbiTokenizer:
    """
    BPE tokenizer wrapping HuggingFace tokenizers with chat special tokens.

    Uses GPT-2 BPE as the base, adds special tokens for chat formatting.
    """

    def __init__(self, base="gpt2"):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_pretrained(base)
        # Record base vocab size before adding specials
        self._base_vocab_size = self.tokenizer.get_vocab_size()
        # Add special tokens
        self.special_ids = {}
        from tokenizers import AddedToken
        for tok in SPECIAL_TOKENS:
            tid = self.tokenizer.get_vocab_size()
            self.tokenizer.add_special_tokens([AddedToken(tok, special=True)])
            self.special_ids[tok] = tid
        self.bos_id = self.special_ids["<|bos|>"]
        self.eos_id = self.special_ids["<|eos|>"]
        self.user_start_id = self.special_ids["<|user_start|>"]
        self.user_end_id = self.special_ids["<|user_end|>"]
        self.assistant_start_id = self.special_ids["<|assistant_start|>"]
        self.assistant_end_id = self.special_ids["<|assistant_end|>"]

    def encode(self, text):
        """Encode text to token ids."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        """Decode token ids to text."""
        return self.tokenizer.decode(ids)

    def get_vocab_size(self):
        """Return total vocab size including special tokens."""
        return self.tokenizer.get_vocab_size()

    def get_vocab_size_padded(self, multiple=64):
        """Return vocab size padded to nearest multiple for tensor core efficiency."""
        v = self.get_vocab_size()
        return ((v + multiple - 1) // multiple) * multiple

    def render_conversation(self, messages):
        """
        Render a conversation to token ids with a loss mask.

        Args:
            messages: list of {"role": "user"|"assistant", "content": str}

        Returns:
            ids: list of int — token ids
            mask: list of int — 1 for tokens where loss should be computed
                  (assistant tokens only), 0 otherwise
        """
        ids = [self.bos_id]
        mask = [0]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            content_ids = self.encode(content)

            if role == "user":
                ids.append(self.user_start_id)
                mask.append(0)
                ids.extend(content_ids)
                mask.extend([0] * len(content_ids))
                ids.append(self.user_end_id)
                mask.append(0)
            elif role == "assistant":
                ids.append(self.assistant_start_id)
                mask.append(0)
                ids.extend(content_ids)
                mask.extend([1] * len(content_ids))  # train on assistant content
                ids.append(self.assistant_end_id)
                mask.append(1)  # train on end token too
            # skip system or other roles

        ids.append(self.eos_id)
        mask.append(0)
        return ids, mask


# Global tokenizer singleton (lazy init)
_tokenizer = None

def get_tokenizer():
    """Get or create the global BPE tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = HebbiTokenizer()
    return _tokenizer


# ---------------------------------------------------------------------------
# Pretrain data loaders (TinyStories, ClimbMix)
# ---------------------------------------------------------------------------

def _load_hf_dataset(dataset_name, split="train", streaming=True):
    """Load a HuggingFace dataset with streaming."""
    from datasets import load_dataset
    return load_dataset(dataset_name, split=split, streaming=streaming)


def pretrain_data_loader(dataset_name, tokenizer, batch_size, seq_len, device,
                         split="train", text_field="text"):
    """
    Infinite iterator for pretraining: stream text from a HF dataset,
    tokenize, and pack into fixed-length sequences.

    Yields (inputs, targets) of shape (B, T) where targets = inputs shifted by 1.
    """
    ds = _load_hf_dataset(dataset_name, split=split)

    # Token buffer — accumulate tokens across documents
    buffer = []
    total_needed = (seq_len + 1) * batch_size  # +1 for targets shift

    ds_iter = iter(ds)
    while True:
        # Fill buffer
        while len(buffer) < total_needed:
            try:
                example = next(ds_iter)
            except StopIteration:
                # Restart dataset
                ds_iter = iter(_load_hf_dataset(dataset_name, split=split))
                example = next(ds_iter)
            text = example.get(text_field, "")
            if text:
                tokens = tokenizer.encode(text)
                buffer.extend(tokens)
                buffer.append(tokenizer.eos_id)

        # Extract batch
        batch_tokens = buffer[:total_needed]
        buffer = buffer[total_needed:]

        # Reshape into (B, T+1), then split
        t = torch.tensor(batch_tokens, dtype=torch.long).view(batch_size, seq_len + 1)
        x = t[:, :seq_len].to(device)
        y = t[:, 1:seq_len + 1].to(device)
        yield x, y


# ---------------------------------------------------------------------------
# SFT data loader (SmolTalk and similar conversation datasets)
# ---------------------------------------------------------------------------

def sft_data_loader(dataset_name, tokenizer, batch_size, seq_len, device,
                    split="train", messages_field="messages"):
    """
    Infinite iterator for SFT: load conversations, render with special tokens,
    and apply loss masking (only train on assistant tokens).

    Yields (inputs, targets, loss_mask) of shape (B, T).
    loss_mask is 1 for positions where the loss should be computed.
    """
    ds = _load_hf_dataset(dataset_name, split=split)

    batch_ids = []
    batch_masks = []
    ds_iter = iter(ds)

    while True:
        # Collect enough conversations for a batch
        while len(batch_ids) < batch_size:
            try:
                example = next(ds_iter)
            except StopIteration:
                ds_iter = iter(_load_hf_dataset(dataset_name, split=split))
                example = next(ds_iter)

            messages = example.get(messages_field, [])
            if not messages:
                continue

            ids, mask = tokenizer.render_conversation(messages)

            # Truncate or skip if too long
            if len(ids) > seq_len + 1:
                ids = ids[:seq_len + 1]
                mask = mask[:seq_len + 1]
            elif len(ids) < 4:
                # Too short to be useful
                continue

            # Pad to seq_len + 1
            pad_len = (seq_len + 1) - len(ids)
            if pad_len > 0:
                ids.extend([tokenizer.eos_id] * pad_len)
                mask.extend([0] * pad_len)

            batch_ids.append(ids)
            batch_masks.append(mask)

        # Extract batch
        cur_ids = batch_ids[:batch_size]
        cur_masks = batch_masks[:batch_size]
        batch_ids = batch_ids[batch_size:]
        batch_masks = batch_masks[batch_size:]

        t_ids = torch.tensor(cur_ids, dtype=torch.long)
        t_mask = torch.tensor(cur_masks, dtype=torch.long)

        # inputs = ids[:-1], targets = ids[1:], mask = mask[1:]
        x = t_ids[:, :seq_len].to(device)
        y = t_ids[:, 1:seq_len + 1].to(device)
        loss_mask = t_mask[:, 1:seq_len + 1].to(device)

        yield x, y, loss_mask


# ---------------------------------------------------------------------------
# Memory training data loader — yields (context, target) pairs from
# split conversations so memory banks get relevant content to store.
# ---------------------------------------------------------------------------

def memory_data_loader(dataset_name, tokenizer, batch_size, seq_len, device,
                       split="train", messages_field="messages",
                       mode="split"):
    """
    Infinite iterator for memory training.

    Modes:
      - "split":  split each conversation at a turn boundary — first half
                  becomes context (Hebbian write), second half becomes target.
      - "replay": same conversation for both context and target.
      - "random": unrelated context and target batches (original behavior).

    Yields (context_ids, target_ids) each of shape (B, T).
    """
    ds = _load_hf_dataset(dataset_name, split=split)
    ds_iter = iter(ds)

    def _next_example():
        nonlocal ds_iter
        while True:
            try:
                example = next(ds_iter)
            except StopIteration:
                ds_iter = iter(_load_hf_dataset(dataset_name, split=split))
                example = next(ds_iter)
            messages = example.get(messages_field, [])
            if messages and len(messages) >= 2:
                return messages

    def _to_tensor(ids_list):
        """Pad/truncate a list of id lists to (B, seq_len) tensor."""
        result = []
        for ids in ids_list:
            if len(ids) > seq_len:
                ids = ids[:seq_len]
            elif len(ids) < seq_len:
                ids = ids + [tokenizer.eos_id] * (seq_len - len(ids))
            result.append(ids)
        return torch.tensor(result, dtype=torch.long, device=device)

    if mode == "random":
        # Original behavior: two independent streams
        ctx_buf = []
        tgt_buf = []
        while True:
            while len(ctx_buf) < batch_size:
                msgs = _next_example()
                ids, _ = tokenizer.render_conversation(msgs)
                ctx_buf.append(ids)
            while len(tgt_buf) < batch_size:
                msgs = _next_example()
                ids, _ = tokenizer.render_conversation(msgs)
                tgt_buf.append(ids)
            yield _to_tensor(ctx_buf[:batch_size]), _to_tensor(tgt_buf[:batch_size])
            ctx_buf = ctx_buf[batch_size:]
            tgt_buf = tgt_buf[batch_size:]

    elif mode == "replay":
        # Same conversation for context and target
        buf = []
        while True:
            while len(buf) < batch_size:
                msgs = _next_example()
                ids, _ = tokenizer.render_conversation(msgs)
                if len(ids) >= 4:
                    buf.append(ids)
            batch = buf[:batch_size]
            buf = buf[batch_size:]
            t = _to_tensor(batch)
            yield t, t.clone()

    elif mode == "split":
        # Split each conversation at a turn boundary
        ctx_buf = []
        tgt_buf = []
        while True:
            while len(ctx_buf) < batch_size:
                msgs = _next_example()
                # Find a split point: half the turns (at least 1 for each half)
                n = len(msgs)
                # Split at the midpoint, rounding to a turn boundary
                # Ensure at least 1 message per half
                split_idx = max(1, n // 2)
                ctx_msgs = msgs[:split_idx]
                tgt_msgs = msgs[split_idx:]
                if not tgt_msgs:
                    # Conversation too short to split; use as replay
                    ids, _ = tokenizer.render_conversation(msgs)
                    if len(ids) >= 4:
                        ctx_buf.append(ids)
                        tgt_buf.append(ids)
                    continue
                ctx_ids, _ = tokenizer.render_conversation(ctx_msgs)
                tgt_ids, _ = tokenizer.render_conversation(tgt_msgs)
                if len(ctx_ids) >= 4 and len(tgt_ids) >= 4:
                    ctx_buf.append(ctx_ids)
                    tgt_buf.append(tgt_ids)
            yield (_to_tensor(ctx_buf[:batch_size]),
                   _to_tensor(tgt_buf[:batch_size]))
            ctx_buf = ctx_buf[batch_size:]
            tgt_buf = tgt_buf[batch_size:]
    else:
        raise ValueError(f"Unknown memory data mode: {mode}")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "shakespeare": {
        "type": "char",
        "description": "Tiny Shakespeare (character-level, ~1M chars)",
    },
    "tinystories": {
        "type": "pretrain",
        "hf_name": "roneneldan/TinyStories",
        "text_field": "text",
        "description": "TinyStories (~2M short stories, ~500MB)",
    },
    "climbmix": {
        "type": "pretrain",
        "hf_name": "karpathy/climbmix-400b-shuffle",
        "text_field": "text",
        "description": "ClimbMix-400B (400B tokens, web+books+code)",
    },
    "smoltalk": {
        "type": "sft",
        "hf_name": "HuggingFaceTB/smol-smoltalk",
        "messages_field": "messages",
        "description": "SmolTalk (~500K conversations)",
    },
}


def get_data_loader(dataset_key, tokenizer, batch_size, seq_len, device,
                    split="train"):
    """
    Create a data loader by dataset key.

    For 'shakespeare': returns char_data_loader (character-level, no tokenizer needed).
    For pretrain datasets: returns pretrain_data_loader.
    For SFT datasets: returns sft_data_loader.
    """
    info = DATASETS[dataset_key]

    if info["type"] == "char":
        text = get_shakespeare()
        ds = CharDataset(text)
        return char_data_loader(ds.data, batch_size, seq_len, device, split), ds

    elif info["type"] == "pretrain":
        loader = pretrain_data_loader(
            info["hf_name"], tokenizer, batch_size, seq_len, device,
            split=split, text_field=info["text_field"],
        )
        return loader, None

    elif info["type"] == "sft":
        loader = sft_data_loader(
            info["hf_name"], tokenizer, batch_size, seq_len, device,
            split=split, messages_field=info["messages_field"],
        )
        return loader, None
