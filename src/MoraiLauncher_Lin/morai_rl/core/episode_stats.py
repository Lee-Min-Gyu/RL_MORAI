from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


EPISODE_CSV_FIELDS = [
    "source",
    "episode_index",
    "total_timesteps",
    "policy_version",
    "worker_id",
    "scenario_name",
    "termination_reason",
    "step_count",
    "episode_progress_m",
    "episode_reward",
    "episode_duration_sec",
    "lap_completed",
    "reward_terms_json",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def signed_reward_terms(reward_terms: dict[str, Any]) -> dict[str, float]:
    signed: dict[str, float] = {}
    for key, value in reward_terms.items():
        if not isinstance(value, (int, float)):
            continue
        signed_value = float(value)
        if key.endswith("_penalty"):
            signed_value = -signed_value
        signed[key] = signed_value
    return signed


def build_episode_summary(
    *,
    source: str,
    episode_index: int,
    info: dict,
    episode_reward: float,
    reward_terms: dict[str, Any] | None = None,
    total_timesteps: int | None = None,
    policy_version: int | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    terms = reward_terms if isinstance(reward_terms, dict) else info.get("reward_terms")
    if not isinstance(terms, dict):
        terms = {}
    return {
        "source": source,
        "episode_index": int(episode_index),
        "total_timesteps": total_timesteps,
        "policy_version": policy_version,
        "worker_id": worker_id,
        "scenario_name": info.get("scenario_name"),
        "termination_reason": info.get("termination_reason"),
        "step_count": safe_int(info.get("step_count")),
        "episode_progress_m": safe_float(info.get("episode_progress_m")),
        "episode_reward": float(episode_reward),
        "episode_duration_sec": safe_float(info.get("episode_duration_sec")),
        "lap_completed": bool(info.get("lap_completed", False)),
        "reward_terms": signed_reward_terms(terms),
    }


class EpisodeStatsWriter:
    def __init__(self, stats_dir: str | Path, prefix: str = "episodes") -> None:
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.stats_dir / f"{prefix}.csv"
        self.jsonl_path = self.stats_dir / f"{prefix}.jsonl"
        self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._jsonl_file = self.jsonl_path.open("a", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=EPISODE_CSV_FIELDS)
        if self.csv_path.stat().st_size == 0:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def write(self, summary: dict[str, Any]) -> None:
        row = {field: summary.get(field) for field in EPISODE_CSV_FIELDS}
        row["reward_terms_json"] = json.dumps(
            summary.get("reward_terms", {}),
            ensure_ascii=True,
            sort_keys=True,
        )
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        self._jsonl_file.write(json.dumps(summary, ensure_ascii=True, sort_keys=True) + "\n")
        self._jsonl_file.flush()

    def close(self) -> None:
        self._csv_file.close()
        self._jsonl_file.close()


class ScenarioStatsAccumulator:
    def __init__(self) -> None:
        self._by_scenario: dict[str, dict[str, Any]] = {}

    def add(self, summary: dict[str, Any]) -> None:
        scenario = str(summary.get("scenario_name") or "unknown")
        stats = self._by_scenario.setdefault(
            scenario,
            {
                "episodes": 0,
                "reward_sum": 0.0,
                "progress_sum": 0.0,
                "steps_sum": 0,
                "lap_completed": 0,
                "reasons": {},
            },
        )
        stats["episodes"] += 1
        stats["reward_sum"] += safe_float(summary.get("episode_reward"))
        stats["progress_sum"] += safe_float(summary.get("episode_progress_m"))
        stats["steps_sum"] += safe_int(summary.get("step_count"))
        if summary.get("lap_completed"):
            stats["lap_completed"] += 1
        reason = str(summary.get("termination_reason") or "unknown")
        reasons = stats["reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1

    def print_summary(self, *, label: str, total_timesteps: int | None = None) -> None:
        if not self._by_scenario:
            return
        step_text = "" if total_timesteps is None else f" total_timesteps={total_timesteps}"
        print(f"{label}{step_text}", flush=True)
        for scenario in sorted(self._by_scenario):
            stats = self._by_scenario[scenario]
            episodes = max(1, int(stats["episodes"]))
            reason_text = ",".join(
                f"{reason}:{count}" for reason, count in sorted(stats["reasons"].items())
            )
            print(
                "  "
                f"scenario={scenario} episodes={episodes} "
                f"reward_mean={stats['reward_sum'] / episodes:+.3f} "
                f"progress_mean={stats['progress_sum'] / episodes:+.2f} "
                f"steps_mean={stats['steps_sum'] / episodes:.1f} "
                f"lap_completed={stats['lap_completed']} "
                f"reasons={reason_text}",
                flush=True,
            )
