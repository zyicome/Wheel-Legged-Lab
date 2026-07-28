#!/usr/bin/env python3

"""Run the complete flat-to-moving-jump RSL-RL training pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("staged_training_config.json")
DEFAULT_ISAACLAB = PROJECT_ROOT.parents[1] / "IsaacLab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--isaaclab-path", type=Path, default=DEFAULT_ISAACLAB)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--flat-checkpoint",
        type=Path,
        default=None,
        help="Skip flat training and start by converting this proven flat checkpoint.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands and gates without training."
    )
    return parser.parse_args()


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("\n[PIPELINE] " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = [
        path for path in run_dir.glob("model_*.pt") if checkpoint_iteration(path) >= 0
    ]
    if not checkpoints:
        raise RuntimeError(f"No model checkpoint was saved in {run_dir}")
    return max(checkpoints, key=checkpoint_iteration)


def find_run_dir(experiment: str, run_name: str) -> Path:
    root = PROJECT_ROOT / "logs" / "rsl_rl" / experiment
    matches = sorted(
        root.glob(f"*_{run_name}"), key=lambda path: path.stat().st_mtime
    )
    if not matches:
        raise RuntimeError(f"Cannot find run '*_{run_name}' below {root}")
    return matches[-1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stages = config["stages"]
    if not stages or stages[0]["name"] != "flat":
        raise ValueError("The first staged-training entry must be 'flat'.")

    isaaclab_sh = args.isaaclab_path.expanduser().resolve() / "isaaclab.sh"
    if not isaaclab_sh.is_file():
        raise FileNotFoundError(f"Cannot find Isaac Lab launcher: {isaaclab_sh}")

    pipeline_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    state_dir = (
        PROJECT_ROOT / "logs" / "rsl_rl" / "staged_pipeline" / pipeline_id
    )
    state = {
        "pipeline_id": pipeline_id,
        "config": str(config_path),
        "completed": [],
        "passed": False,
    }
    if not args.dry_run:
        write_json(state_dir / "state.json", state)

    previous_checkpoint: Path | None = None
    first_stage_index = 0
    if args.flat_checkpoint is not None:
        previous_checkpoint = args.flat_checkpoint.expanduser().resolve()
        if not previous_checkpoint.is_file():
            raise FileNotFoundError(f"Flat checkpoint does not exist: {previous_checkpoint}")
        first_stage_index = 1
        converted = state_dir / "model_flat_jump_init.pt"
        conversion_command = [
            str(isaaclab_sh),
            "-p",
            "scripts/expand_rsl_checkpoint_for_jump.py",
            str(previous_checkpoint),
            "--output",
            str(converted),
        ]
        run_command(conversion_command, dry_run=args.dry_run)
        previous_checkpoint = (
            Path("<converted-flat-checkpoint>") if args.dry_run else converted
        )
        if not args.dry_run:
            state["source_flat_checkpoint"] = str(args.flat_checkpoint)
            state["flat_jump_checkpoint"] = str(converted)
            write_json(state_dir / "state.json", state)

    for index, stage in enumerate(stages):
        if index < first_stage_index:
            continue

        stage_name = stage["name"]
        run_name = f"auto_{pipeline_id}_{stage_name}"
        gate_path = state_dir / f"{index:02d}_{stage_name}_gate.json"
        status_path = state_dir / f"{index:02d}_{stage_name}_status.json"
        gate = {
            key: stage[key]
            for key in ("name", "min_iterations", "window", "consecutive", "rules")
        }
        if not args.dry_run:
            write_json(gate_path, gate)

        command = [
            str(isaaclab_sh),
            "-p",
            "scripts/rsl_rl/train.py",
            "--task",
            stage["task"],
            "--headless",
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(stage["max_iterations"]),
            "--run_name",
            run_name,
            "--early_stop_config",
            str(gate_path),
            "--early_stop_status_file",
            str(status_path),
        ]
        if args.device is not None:
            command.extend(("--device", args.device))
        if args.seed is not None:
            command.extend(("--seed", str(args.seed)))
        if previous_checkpoint is not None:
            command.extend(
                (
                    "--load_checkpoint_path",
                    str(previous_checkpoint),
                    "--load_weights_only",
                )
            )

        run_command(command, dry_run=args.dry_run)
        if args.dry_run:
            previous_checkpoint = Path(f"<checkpoint-from-{stage_name}>")
            continue

        run_dir = find_run_dir(stage["experiment"], run_name)
        checkpoint = latest_checkpoint(run_dir)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        state["active_stage"] = stage_name
        state["last_checkpoint"] = str(checkpoint)
        state["last_status"] = status
        write_json(state_dir / "state.json", state)
        if not status.get("passed", False):
            print(
                f"\n[PIPELINE] Stage {stage_name!r} exhausted its training budget "
                f"without satisfying the gate.\n"
                f"[PIPELINE] Checkpoint retained at: {checkpoint}\n"
                f"[PIPELINE] Status: {status_path}",
                file=sys.stderr,
            )
            return 2

        state["completed"].append(
            {
                "stage": stage_name,
                "task": stage["task"],
                "checkpoint": str(checkpoint),
                "status": status,
            }
        )
        write_json(state_dir / "state.json", state)
        previous_checkpoint = checkpoint

        if stage_name == "flat":
            converted = state_dir / "model_flat_jump_init.pt"
            conversion_command = [
                str(isaaclab_sh),
                "-p",
                "scripts/expand_rsl_checkpoint_for_jump.py",
                str(checkpoint),
                "--output",
                str(converted),
            ]
            run_command(conversion_command, dry_run=False)
            previous_checkpoint = converted
            state["flat_jump_checkpoint"] = str(converted)
            write_json(state_dir / "state.json", state)

    if args.dry_run:
        print("\n[PIPELINE] Dry run completed.")
        return 0

    state["passed"] = True
    state["final_checkpoint"] = str(previous_checkpoint)
    state.pop("active_stage", None)
    write_json(state_dir / "state.json", state)
    print(f"\n[PIPELINE] All stages passed.")
    print(f"[PIPELINE] Final checkpoint: {previous_checkpoint}")
    print(f"[PIPELINE] Pipeline record: {state_dir / 'state.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
