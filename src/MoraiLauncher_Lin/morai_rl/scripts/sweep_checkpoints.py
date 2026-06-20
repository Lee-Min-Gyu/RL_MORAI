from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "stage1_ros_sync_config.toml"
CHECKPOINT_STEP_RE = re.compile(r"ppo_checkpoint_(\d+)_steps(?:\.zip)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple PPO checkpoints and copy the best one by scenario metrics."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--scenario-repeats", type=int, default=3)
    parser.add_argument("--scenarios", default="")
    parser.add_argument(
        "--every-steps",
        type=int,
        default=50_000,
        help="Evaluate checkpoints nearest to this step interval. 0 evaluates all checkpoints.",
    )
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--max-step", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--disable-map-plots", action="store_true")
    parser.add_argument("--best-name", default="best_checkpoint.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"checkpoint dir not found: {checkpoint_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir.parent / "checkpoint_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _select_checkpoints(
        checkpoint_dir=checkpoint_dir,
        every_steps=int(args.every_steps),
        min_step=int(args.min_step),
        max_step=int(args.max_step),
        limit=int(args.limit),
    )
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints selected from {checkpoint_dir}")

    print(f"sweep_output_dir={output_dir}", flush=True)
    print("sweep_selected_checkpoints", flush=True)
    for checkpoint in checkpoints:
        print(f"  step={checkpoint.step} path={checkpoint.path}", flush=True)

    results: list[dict] = []
    for checkpoint in checkpoints:
        eval_dir = output_dir / f"checkpoint_{checkpoint.step}_steps"
        eval_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "morai_rl.scripts.evaluate_ppo",
            "--config",
            str(args.config),
            "--model",
            str(checkpoint.path),
            "--stats-dir",
            str(eval_dir),
            "--scenario-repeats",
            str(max(1, int(args.scenario_repeats))),
        ]
        if args.scenarios.strip():
            command.extend(["--scenarios", args.scenarios])
        if args.stochastic:
            command.append("--stochastic")
        if args.disable_map_plots:
            command.append("--disable-map-plots")
        print(f"sweep_eval_start step={checkpoint.step}", flush=True)
        subprocess.run(command, check=True)
        metrics = _score_eval_csv(eval_dir / "eval_episodes.csv")
        metrics["step"] = checkpoint.step
        metrics["checkpoint_path"] = str(checkpoint.path)
        metrics["eval_dir"] = str(eval_dir)
        results.append(metrics)
        print(
            "sweep_eval_result "
            f"step={checkpoint.step} lap_completed={metrics['lap_completed']} "
            f"min_scenario_progress_mean={metrics['min_scenario_progress_mean']:+.2f} "
            f"avg_progress={metrics['avg_progress']:+.2f} "
            f"avg_reward={metrics['avg_reward']:+.3f}",
            flush=True,
        )

    _write_sweep_summary(output_dir / "sweep_summary.csv", results)
    best = max(results, key=_score_key)
    best_source = Path(str(best["checkpoint_path"]))
    best_target = output_dir / args.best_name
    shutil.copy2(best_source, best_target)
    print(
        "sweep_best_checkpoint "
        f"step={best['step']} source={best_source} copied_to={best_target}",
        flush=True,
    )


class SelectedCheckpoint:
    def __init__(self, step: int, path: Path) -> None:
        self.step = int(step)
        self.path = path


def _select_checkpoints(
    *,
    checkpoint_dir: Path,
    every_steps: int,
    min_step: int,
    max_step: int,
    limit: int,
) -> list[SelectedCheckpoint]:
    found: list[SelectedCheckpoint] = []
    for path in checkpoint_dir.glob("ppo_checkpoint_*_steps.zip"):
        match = CHECKPOINT_STEP_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if min_step > 0 and step < min_step:
            continue
        if max_step > 0 and step > max_step:
            continue
        found.append(SelectedCheckpoint(step=step, path=path))
    found.sort(key=lambda item: item.step)
    if every_steps <= 0:
        selected = found
    else:
        selected_by_bucket: dict[int, SelectedCheckpoint] = {}
        for checkpoint in found:
            bucket = round(checkpoint.step / every_steps)
            current = selected_by_bucket.get(bucket)
            target_step = bucket * every_steps
            if current is None or abs(checkpoint.step - target_step) < abs(current.step - target_step):
                selected_by_bucket[bucket] = checkpoint
        selected = sorted(selected_by_bucket.values(), key=lambda item: item.step)
    if limit > 0:
        selected = selected[-limit:]
    return selected


def _score_eval_csv(path: Path) -> dict:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if not rows:
        raise ValueError(f"empty eval csv: {path}")
    by_scenario: dict[str, list[dict]] = {}
    for row in rows:
        by_scenario.setdefault(str(row.get("scenario_name") or "unknown"), []).append(row)
    progress_means = []
    for scenario_rows in by_scenario.values():
        progress_means.append(
            sum(_float(row.get("episode_progress_m")) for row in scenario_rows) / len(scenario_rows)
        )
    return {
        "episodes": len(rows),
        "scenario_count": len(by_scenario),
        "lap_completed": sum(1 for row in rows if str(row.get("lap_completed")).lower() == "true"),
        "min_scenario_progress_mean": min(progress_means) if progress_means else 0.0,
        "avg_progress": sum(_float(row.get("episode_progress_m")) for row in rows) / len(rows),
        "avg_reward": sum(_float(row.get("episode_reward")) for row in rows) / len(rows),
        "avg_steps": sum(_float(row.get("step_count")) for row in rows) / len(rows),
    }


def _score_key(metrics: dict) -> tuple:
    return (
        int(metrics["lap_completed"]),
        float(metrics["min_scenario_progress_mean"]),
        float(metrics["avg_progress"]),
        float(metrics["avg_reward"]),
        int(metrics["step"]),
    )


def _write_sweep_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "step",
        "checkpoint_path",
        "eval_dir",
        "episodes",
        "scenario_count",
        "lap_completed",
        "min_scenario_progress_mean",
        "avg_progress",
        "avg_reward",
        "avg_steps",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["step"])):
            writer.writerow({field: row.get(field) for field in fields})
    print(f"saved_sweep_summary={path}", flush=True)


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
