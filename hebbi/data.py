"""
Character-level data loading for DET.
Downloads tinyshakespeare and provides simple infinite data iterators.
"""

import os
import urllib.request
import torch


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
