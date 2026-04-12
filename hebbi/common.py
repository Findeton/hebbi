"""
Common utilities for DET. Adapted from nanochat.
"""

import os
import torch

# COMPUTE_DTYPE detection (same pattern as nanochat)
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

def _detect_compute_dtype():
    env = os.environ.get("HEBBI_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via HEBBI_DTYPE={env}"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16)"
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"

COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()


def print0(s="", **kwargs):
    print(s, **kwargs)


def autodetect_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def compute_init(device_type="cuda"):
    """Basic initialization: seeds, precision, device."""
    torch.manual_seed(42)
    if device_type == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
    elif device_type == "mps":
        assert torch.backends.mps.is_available()
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print0(f"Device: {device_type} | COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    return device


class DummyWandb:
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
