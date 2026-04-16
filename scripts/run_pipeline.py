"""
Hebbi training pipeline runner — resumable end-to-end training.

Runs all stages in sequence:
    1. TinyStories pretrain       (20k steps)
    2. ClimbMix pretrain          (50k steps, weights initialized from stage 1)
    3. SmolTalk SFT               (3k steps, from stage 2)
    4. Energy consolidation       (2k steps, from stage 3, energy_steps=3)
    5. Memory gate training       (1k steps, from stage 4)

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

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
pipeline_parser = argparse.ArgumentParser(description="Hebbi training pipeline")
pipeline_parser.add_argument("--backprop", action="store_true",
                             help="use standard backprop instead of FF (baseline)")
pipeline_parser.add_argument("--restart-from", type=str, default=None,
                             choices=["tinystories", "climbmix", "sft", "energy", "memory"],
                             help="restart pipeline from this stage (clears it and all later stages)")
pipeline_args = pipeline_parser.parse_args()


# ---------------------------------------------------------------------------
# Paths — backprop runs get their own directory to avoid colliding with FF
# ---------------------------------------------------------------------------
_default_runs = "/workspace/runs_backprop" if pipeline_args.backprop else "/workspace/runs"
RUNS_DIR = Path(os.environ.get("HEBBI_RUNS_DIR", _default_runs))
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
            "--batch-size=8",
            "--grad-accum=6",
            "--num-iterations=20000",
            "--warmup-steps=1000",
            "--block-lr=3e-4",
            "--head-lr=3e-4",
            "--embed-lr=3e-3",
            "--save-every=2500",
            "--sample-every=2000",
            "--eval-every=-1",
            "--device-type=cuda",
            "--adaptive-threshold",
            "--predictive-negatives",
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
            "--batch-size=8",
            "--grad-accum=12",
            "--num-iterations=50000",
            "--warmup-steps=1500",
            "--warmdown-ratio=0.65",
            "--block-lr=3e-4",
            "--head-lr=3e-4",
            "--embed-lr=3e-3",
            "--save-every=2500",
            "--sample-every=5000",
            "--adaptive-threshold",
            "--predictive-negatives",
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
            "--batch-size=4",
            "--num-iterations=3000",
            "--block-lr=3e-4",
            "--head-lr=3e-4",
            "--embed-lr=1e-3",
            "--warmup-steps=50",
            "--save-every=1000",
            "--sample-every=500",
        ],
    },
    {
        "name": "energy",
        "script": "scripts.train_energy",
        "num_iterations": 2000,
        "init_from": "sft",
        "checkpoint_flag": "--checkpoint",
        "final_name": "hebbi_energy_final.pt",
        "checkpoint_prefix": "hebbi_energy_",
        "args": [
            "--dataset=smoltalk",
            "--batch-size=4",
            "--num-iterations=2000",
            "--energy-steps=3",
            "--energy-weight-mode=increasing",
            "--ilsd-weight=0.3",
            "--block-lr=1e-4",
            "--head-lr=1e-4",
            "--embed-lr=3e-4",
            "--warmup-steps=100",
            "--save-every=500",
            "--sample-every=500",
        ],
    },
    {
        "name": "memory",
        "script": "scripts.train_memory",
        "num_iterations": 1000,
        "init_from": "energy",
        "checkpoint_flag": "--checkpoint",
        "final_name": "hebbi_memory_final.pt",
        "checkpoint_prefix": "hebbi_memory_",
        "args": [
            "--dataset=smoltalk",
            "--batch-size=4",
            "--num-iterations=1000",
            "--block-lr=3e-4",
            "--head-lr=3e-4",
            "--embed-lr=1e-3",
            "--warmup-steps=50",
            "--save-every=500",
            "--sample-every=500",
        ],
    },
]


# ---------------------------------------------------------------------------
# Backprop overrides: ~5x fewer iterations, checkpoints every 500 steps
# ---------------------------------------------------------------------------
if pipeline_args.backprop:
    _BACKPROP_OVERRIDES = {
        "tinystories":  {"--num-iterations": "4000",  "--save-every": "500",
                         "--sample-every": "500",  "--warmup-steps": "200"},
        "climbmix":     {"--num-iterations": "40000", "--save-every": "500",
                         "--sample-every": "500", "--warmup-steps": "1000"},
        "sft":          {"--num-iterations": "3000",  "--save-every": "500",
                         "--sample-every": "200",  "--warmup-steps": "50",
                         "--block-lr": "5e-5", "--head-lr": "5e-5",
                         "--embed-lr": "1e-4"},
        "energy":       {"--num-iterations": "400",   "--save-every": "200",
                         "--sample-every": "200",  "--warmup-steps": "20",
                         "--batch-size": "2"},
        "memory":       {"--num-iterations": "500",   "--save-every": "100",
                         "--sample-every": "100",  "--warmup-steps": "20"},
    }
    for stage in STAGES:
        overrides = _BACKPROP_OVERRIDES.get(stage["name"], {})
        if overrides:
            # Replace matching args in-place
            new_args = []
            for arg in stage["args"]:
                key = arg.split("=")[0] if "=" in arg else arg
                if key in overrides:
                    new_args.append(f"{key}={overrides[key]}")
                    del overrides[key]
                else:
                    new_args.append(arg)
            # Append any remaining overrides not already present
            for key, val in overrides.items():
                new_args.append(f"{key}={val}")
            stage["args"] = new_args
            # Update num_iterations metadata too
            if "--num-iterations" in _BACKPROP_OVERRIDES.get(stage["name"], {}):
                stage["num_iterations"] = int(
                    _BACKPROP_OVERRIDES[stage["name"]]["--num-iterations"])

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

    stage_args = list(stage["args"])

    # Backprop mode: add --backprop flag and strip FF-only flags
    if pipeline_args.backprop:
        stage_args = [a for a in stage_args
                      if a not in ("--adaptive-threshold", "--predictive-negatives")]
        stage_args.append("--backprop")

    cmd = [sys.executable, "-m", stage["script"]] + stage_args
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

    # --restart-from: clear this stage and all later stages so they re-run
    if pipeline_args.restart_from is not None:
        restart = pipeline_args.restart_from
        stage_names = [s["name"] for s in STAGES]
        if restart not in stage_names:
            log(f"Unknown stage: {restart}")
            sys.exit(1)
        restart_idx = stage_names.index(restart)
        stages_to_clear = stage_names[restart_idx:]
        cleared = []
        for sname in stages_to_clear:
            if sname in state["completed"]:
                state["completed"].remove(sname)
                cleared.append(sname)
            # Remove old checkpoints so the stage starts fresh from prev stage
            sd = stage_dir(sname)
            ckpt_d = stage_ckpt_dir(sname)
            if ckpt_d.exists():
                import shutil
                shutil.rmtree(ckpt_d)
                log(f"Cleared checkpoints: {ckpt_d}")
            # Also remove init_from_prev.pt so it gets re-created
            init_ckpt = sd / "init_from_prev.pt"
            if init_ckpt.exists():
                init_ckpt.unlink()
        save_state(state)
        if cleared:
            log(f"Restarting from {restart}: cleared {cleared}")
        else:
            log(f"Restarting from {restart} (none were marked complete)")

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
