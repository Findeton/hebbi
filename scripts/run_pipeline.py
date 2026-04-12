"""
Hebbi training pipeline runner — resumable end-to-end training.

Runs all three stages in sequence:
    1. TinyStories pretrain (20k steps)
    2. ClimbMix pretrain   (50k steps, weights initialized from stage 1)
    3. SmolTalk SFT         (3k steps, from stage 2)

Resumable: if the script is stopped (Ctrl+C, pod restart, crash, etc.),
just re-run it. It:
    - Skips stages already marked as complete in pipeline_state.json
    - For the in-progress stage, resumes from the latest checkpoint on disk

Each stage has its own directory under /workspace/runs/ so their
checkpoints never collide.

Bootstrap (one time, on the RunPod pod):

    cd /workspace
    git clone https://github.com/Findeton/hebbi.git
    cd hebbi
    pip install -e .
    export HF_HOME=/workspace/hf_cache    # persist dataset cache on volume
    python scripts/run_pipeline.py

On resume (just re-run the same command):

    cd /workspace/hebbi
    export HF_HOME=/workspace/hf_cache
    python scripts/run_pipeline.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUNS_DIR = Path(os.environ.get("HEBBI_RUNS_DIR", "/workspace/runs"))
STATE_FILE = RUNS_DIR / "pipeline_state.json"


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
# Each stage runs with cwd set to its own subdirectory so that
# `checkpoints/hebbi_*.pt` files land in stage-scoped dirs.
STAGES = [
    {
        "name": "tinystories",
        "script": "scripts.train",
        "num_iterations": 20000,
        "init_from": None,
        "checkpoint_flag": "--resume",
        "final_name": "hebbi_final.pt",
        "checkpoint_prefix": "hebbi_",
        "args": [
            "--dataset=tinystories",
            "--depth=12",
            "--seq-len=1024",
            "--batch-size=24",
            "--grad-accum=2",
            "--num-iterations=20000",
            "--warmup-steps=200",
            "--save-every=2500",
            "--sample-every=2000",
            "--eval-every=-1",
            "--device-type=cuda",
        ],
    },
    {
        "name": "climbmix",
        "script": "scripts.train",
        "num_iterations": 50000,
        "init_from": "tinystories",
        "checkpoint_flag": "--resume",
        "final_name": "hebbi_final.pt",
        "checkpoint_prefix": "hebbi_",
        "args": [
            "--dataset=climbmix",
            "--depth=12",
            "--seq-len=1024",
            "--batch-size=24",
            "--grad-accum=4",
            "--num-iterations=50000",
            "--warmup-steps=500",
            "--warmdown-ratio=0.65",
            "--save-every=2500",
            "--sample-every=5000",
            "--eval-every=-1",
            "--device-type=cuda",
        ],
    },
    {
        "name": "sft",
        "script": "scripts.train_sft",
        "num_iterations": 3000,
        "init_from": "climbmix",
        "checkpoint_flag": "--checkpoint",
        "final_name": "hebbi_sft_final.pt",
        "checkpoint_prefix": "hebbi_sft_",
        "args": [
            "--dataset=smoltalk",
            "--batch-size=12",
            "--num-iterations=3000",
            "--block-lr=3e-4",
            "--head-lr=3e-4",
            "--embed-lr=1e-3",
            "--warmup-steps=50",
            "--save-every=1000",
            "--sample-every=500",
        ],
    },
]

STAGE_BY_NAME = {s["name"]: s for s in STAGES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(f"[pipeline] {msg}", flush=True)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed": []}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def stage_dir(stage_name):
    return RUNS_DIR / f"stage_{stage_name}"


def stage_ckpt_dir(stage_name):
    return stage_dir(stage_name) / "checkpoints"


def final_checkpoint(stage_name):
    """Return the stage's final checkpoint path, or None if not yet produced."""
    stage = STAGE_BY_NAME[stage_name]
    p = stage_ckpt_dir(stage_name) / stage["final_name"]
    return p if p.exists() else None


def latest_intermediate(stage_name):
    """Return the latest numbered intermediate checkpoint for a stage, or None."""
    stage = STAGE_BY_NAME[stage_name]
    prefix = stage["checkpoint_prefix"]
    ckpt_dir = stage_ckpt_dir(stage_name)
    if not ckpt_dir.exists():
        return None
    best_step = -1
    best_path = None
    for p in ckpt_dir.glob(f"{prefix}*.pt"):
        # Skip the "final" checkpoint and our own init_from_prev file.
        if "final" in p.stem:
            continue
        # Extract the trailing number: e.g. hebbi_002500 → 2500
        tail = p.stem[len(prefix):]
        if not tail.isdigit():
            continue
        step = int(tail)
        if step > best_step:
            best_step = step
            best_path = p
    return best_path


def prepare_init_from_prev(src, dst):
    """
    Copy a previous stage's final checkpoint, but reset the step counter to 0
    so `--resume` starts fresh with a new LR schedule.
    """
    import torch
    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    new_ckpt = {
        "config": ckpt["config"],
        "model": ckpt["model"],
        "step": 0,
    }
    torch.save(new_ckpt, dst)


def build_stage_command(stage):
    """
    Decide which checkpoint to pass and build the argv for the stage.
    Returns (cmd_list, cwd_path).
    """
    name = stage["name"]
    sd = stage_dir(name)
    sd.mkdir(parents=True, exist_ok=True)
    stage_ckpt_dir(name).mkdir(parents=True, exist_ok=True)

    # Priority 1: mid-stage resume from the latest intermediate checkpoint
    latest = latest_intermediate(name)
    ckpt_path = None

    if latest is not None:
        log(f"{name}: resuming from {latest.name}")
        ckpt_path = latest
    elif stage["init_from"] is not None:
        # First run of this stage — bootstrap weights from the previous stage.
        prev = stage["init_from"]
        prev_final = final_checkpoint(prev)
        if prev_final is None:
            raise RuntimeError(
                f"{name}: cannot init from '{prev}' — no final checkpoint "
                f"found at {stage_ckpt_dir(prev) / STAGE_BY_NAME[prev]['final_name']}"
            )

        if stage["script"] == "scripts.train_sft":
            # train_sft.py doesn't look at the 'step' field, pass the file directly.
            ckpt_path = prev_final
            log(f"{name}: initializing from {prev}/{prev_final.name}")
        else:
            # train.py respects the step field — strip it so training starts at step 0.
            init_ckpt = sd / "init_from_prev.pt"
            if not init_ckpt.exists():
                log(f"{name}: preparing init checkpoint from {prev}/{prev_final.name}")
                prepare_init_from_prev(prev_final, init_ckpt)
            ckpt_path = init_ckpt
    else:
        log(f"{name}: starting from scratch")

    cmd = [sys.executable, "-m", stage["script"]] + list(stage["args"])
    if ckpt_path is not None:
        cmd.append(f"{stage['checkpoint_flag']}={ckpt_path}")

    return cmd, sd


def run_stage(stage):
    name = stage["name"]
    cmd, cwd = build_stage_command(stage)

    log(f"=== Stage: {name} ===")
    log(f"cwd:  {cwd}")
    log(f"cmd:  {' '.join(str(c) for c in cmd)}")
    print()

    env = os.environ.copy()
    # Redirect HF cache to the persistent volume when running on RunPod.
    # Only auto-set HF_HOME if /workspace exists (i.e., we're on RunPod or
    # a similar persistent-volume setup). Respect a pre-set HF_HOME always.
    if "HF_HOME" not in env and Path("/workspace").is_dir():
        env["HF_HOME"] = "/workspace/hf_cache"
    if "HF_HOME" in env:
        try:
            Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log(f"warning: could not create HF_HOME={env['HF_HOME']}: {e}")

    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        log(f"Stage {name} exited with code {result.returncode}")
        sys.exit(result.returncode)

    # Verify the final checkpoint landed where expected.
    fp = final_checkpoint(name)
    if fp is None:
        log(f"Stage {name} finished but {stage['final_name']} not found — not marking complete.")
        sys.exit(1)

    log(f"Stage {name} complete. Final: {fp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    log(f"Runs dir: {RUNS_DIR}")
    log(f"State: {state}")
    print()

    for stage in STAGES:
        name = stage["name"]
        if name in state["completed"]:
            log(f"Skipping {name} (already complete)")
            continue

        run_stage(stage)

        state["completed"].append(name)
        save_state(state)
        print()

    log("=== All stages complete ===")
    last_stage = STAGES[-1]["name"]
    final = final_checkpoint(last_stage)
    if final is not None:
        log(f"Final model ({last_stage}): {final}")
        log(f"Size: {final.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted. Re-run this script to resume from the latest checkpoint.")
        sys.exit(130)
