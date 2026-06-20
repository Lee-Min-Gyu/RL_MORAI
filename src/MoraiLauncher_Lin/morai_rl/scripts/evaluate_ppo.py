from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from stable_baselines3 import PPO
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    PPO = None
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None

from morai_rl.config.runtime import load_config
from morai_rl.core.episode_stats import (
    EpisodeStatsWriter,
    ScenarioStatsAccumulator,
    build_episode_summary,
)
from morai_rl.envs.gym_wrapper import GymMoraiEnv
from morai_rl.maps.reference_path import ReferencePath
from morai_rl.maps.route_corridor import RouteCorridor

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "stage1_ros_sync_config.toml"

STEP_TRACE_FIELDS = [
    "episode_index",
    "scenario_name",
    "step",
    "x",
    "y",
    "yaw_deg",
    "speed_mps",
    "throttle_brake_action",
    "steering_action",
    "throttle",
    "brake",
    "progress_m",
    "progress_delta_m",
    "lateral_error_m",
    "heading_error_rad",
    "corridor_distance_m",
    "lookahead_5m_x",
    "lookahead_5m_y",
    "lookahead_5m_heading_error",
    "lookahead_10m_x",
    "lookahead_10m_y",
    "lookahead_10m_heading_error",
    "lookahead_15m_x",
    "lookahead_15m_y",
    "lookahead_15m_heading_error",
    "lookahead_20m_x",
    "lookahead_20m_y",
    "lookahead_20m_heading_error",
    "lookahead_25m_x",
    "lookahead_25m_y",
    "lookahead_25m_heading_error",
    "lookahead_30m_x",
    "lookahead_30m_y",
    "lookahead_30m_heading_error",
    "reward",
    "terminated",
    "truncated",
    "termination_reason",
    "reward_terms_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy in MORAI.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", required=True, help="Path to a PPO .zip checkpoint or path without .zip suffix.")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes. Defaults to scenario count or 5.")
    parser.add_argument("--scenarios", default="", help="Comma-separated scenario names. Defaults to config list.")
    parser.add_argument(
        "--scenario-repeats",
        type=int,
        default=0,
        help="Run each selected scenario this many times. Overrides --episodes when > 0.",
    )
    parser.add_argument("--stats-dir", default="", help="Directory for eval CSV/JSONL logs.")
    parser.add_argument("--stats-prefix", default="eval_episodes")
    parser.add_argument("--disable-step-trace", action="store_true")
    parser.add_argument("--disable-map-plots", action="store_true")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument(
        "--scenario-selection-mode",
        choices=["config", "fixed", "round_robin", "random"],
        default="config",
        help="Override config reset scenario selection during evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    if PPO is None:
        raise ModuleNotFoundError(
            "stable-baselines3 is required. Install it with `pip install stable-baselines3`."
        ) from _SB3_IMPORT_ERROR

    args = parse_args()
    model_path = _resolve_model_path(args.model)
    stats_dir = Path(args.stats_dir) if args.stats_dir else Path("runs/ppo_eval") / model_path.stem
    stats_dir.mkdir(parents=True, exist_ok=True)
    _remove_existing_eval_outputs(stats_dir, args.stats_prefix)
    print(f"eval_model={model_path}", flush=True)
    print(f"eval_stats_dir={stats_dir}", flush=True)

    app_config = load_config(args.config)
    scenarios = _resolve_scenarios(args.scenarios, app_config.reset.scenario_load_file_names)
    eval_plan = _build_eval_plan(
        scenarios=scenarios,
        scenario_repeats=int(args.scenario_repeats),
        episodes=int(args.episodes),
    )
    episodes = len(eval_plan)

    env = GymMoraiEnv(args.config)
    if args.scenario_selection_mode != "config":
        env.env.reset_manager.scenario_selection_mode = args.scenario_selection_mode
        print(f"eval_scenario_selection_mode={args.scenario_selection_mode}", flush=True)

    model = PPO.load(str(model_path), env=env, device="auto")
    writer = EpisodeStatsWriter(stats_dir, prefix=args.stats_prefix)
    scenario_stats = ScenarioStatsAccumulator()
    trace_writer = None if args.disable_step_trace else StepTraceWriter(stats_dir, prefix="step_trace")

    try:
        for episode_index, scenario_name in enumerate(eval_plan, start=1):
            if scenario_name:
                env.env.reset_manager.scenario_file_names = [scenario_name]
                env.env.reset_manager.scenario_selection_mode = "fixed"
            obs, info = env.reset()
            episode_reward = 0.0
            reward_terms: dict[str, float] = {}
            done = False
            last_info = info
            while not done:
                action, _ = model.predict(obs, deterministic=bool(args.deterministic))
                obs, reward, terminated, truncated, last_info = env.step(action)
                episode_reward += float(reward)
                _accumulate_reward_terms(reward_terms, last_info.get("reward_terms"))
                if trace_writer is not None:
                    trace_writer.write(
                        episode_index=episode_index,
                        action=action,
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info=last_info,
                    )
                done = bool(terminated or truncated)

            summary = build_episode_summary(
                source="eval",
                episode_index=episode_index,
                info=last_info,
                episode_reward=episode_reward,
                reward_terms=reward_terms,
                total_timesteps=None,
            )
            writer.write(summary)
            scenario_stats.add(summary)
            print(
                "eval_episode_end "
                f"episode={episode_index} "
                f"scenario={summary.get('scenario_name')} "
                f"reason={summary.get('termination_reason')} "
                f"steps={summary.get('step_count')} "
                f"progress_m={float(summary.get('episode_progress_m', 0.0)):+.2f} "
                f"reward={float(summary.get('episode_reward', 0.0)):+.3f}",
                flush=True,
            )

        scenario_stats.print_summary(label="eval_scenario_stats")
        if not args.disable_map_plots and trace_writer is not None:
            trace_writer.flush()
            _write_global_map_plots(
                config_path=Path(args.config),
                stats_dir=stats_dir,
                trace_csv_path=trace_writer.csv_path,
            )
    finally:
        if trace_writer is not None:
            trace_writer.close()
        writer.close()
        env.close()


def _resolve_model_path(model: str) -> Path:
    path = Path(model)
    if path.is_file():
        return path
    if path.suffix != ".zip" and path.with_suffix(".zip").is_file():
        return path.with_suffix(".zip")
    raise FileNotFoundError(f"model not found: {model}")


def _accumulate_reward_terms(target: dict[str, float], reward_terms) -> None:
    if not isinstance(reward_terms, dict):
        return
    for key, value in reward_terms.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0.0) + float(value)


def _resolve_scenarios(raw_scenarios: str, config_scenarios: list[str]) -> list[str]:
    if raw_scenarios.strip():
        return [item.strip() for item in raw_scenarios.split(",") if item.strip()]
    seen: set[str] = set()
    scenarios: list[str] = []
    for scenario in config_scenarios:
        scenario = str(scenario).strip()
        if scenario and scenario not in seen:
            scenarios.append(scenario)
            seen.add(scenario)
    return scenarios


def _build_eval_plan(scenarios: list[str], scenario_repeats: int, episodes: int) -> list[str | None]:
    if scenario_repeats > 0:
        if not scenarios:
            raise ValueError("--scenario-repeats requires at least one scenario")
        return [
            scenario
            for scenario in scenarios
            for _ in range(int(scenario_repeats))
        ]
    episode_count = episodes if episodes > 0 else max(5, len(scenarios))
    if scenarios:
        return [scenarios[index % len(scenarios)] for index in range(episode_count)]
    return [None for _ in range(episode_count)]


class StepTraceWriter:
    def __init__(self, stats_dir: str | Path, prefix: str = "step_trace") -> None:
        self.csv_path = Path(stats_dir) / f"{prefix}.csv"
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=STEP_TRACE_FIELDS)
        self._writer.writeheader()

    def write(
        self,
        *,
        episode_index: int,
        action,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> None:
        action_values = list(float(value) for value in getattr(action, "ravel", lambda: action)())
        state = info.get("state") or {}
        projection = info.get("projection") or {}
        corridor = info.get("corridor") or {}
        observation_named = info.get("observation_named") or {}
        reward_terms = info.get("reward_terms") if isinstance(info.get("reward_terms"), dict) else {}
        self._writer.writerow(
            {
                "episode_index": int(episode_index),
                "scenario_name": info.get("scenario_name"),
                "step": info.get("step_count"),
                "x": state.get("x"),
                "y": state.get("y"),
                "yaw_deg": state.get("yaw_deg"),
                "speed_mps": state.get("speed_mps"),
                "throttle_brake_action": action_values[0] if len(action_values) > 0 else None,
                "steering_action": action_values[1] if len(action_values) > 1 else None,
                "throttle": state.get("throttle"),
                "brake": state.get("brake"),
                "progress_m": info.get("episode_progress_m"),
                "progress_delta_m": info.get("progress_delta_m"),
                "lateral_error_m": projection.get("lateral_error_m"),
                "heading_error_rad": projection.get("heading_error_rad"),
                "corridor_distance_m": corridor.get("corridor_distance_m"),
                "lookahead_5m_x": observation_named.get("lookahead_5m_x"),
                "lookahead_5m_y": observation_named.get("lookahead_5m_y"),
                "lookahead_5m_heading_error": observation_named.get("lookahead_5m_heading_error"),
                "lookahead_10m_x": observation_named.get("lookahead_10m_x"),
                "lookahead_10m_y": observation_named.get("lookahead_10m_y"),
                "lookahead_10m_heading_error": observation_named.get("lookahead_10m_heading_error"),
                "lookahead_15m_x": observation_named.get("lookahead_15m_x"),
                "lookahead_15m_y": observation_named.get("lookahead_15m_y"),
                "lookahead_15m_heading_error": observation_named.get("lookahead_15m_heading_error"),
                "lookahead_20m_x": observation_named.get("lookahead_20m_x"),
                "lookahead_20m_y": observation_named.get("lookahead_20m_y"),
                "lookahead_20m_heading_error": observation_named.get("lookahead_20m_heading_error"),
                "lookahead_25m_x": observation_named.get("lookahead_25m_x"),
                "lookahead_25m_y": observation_named.get("lookahead_25m_y"),
                "lookahead_25m_heading_error": observation_named.get("lookahead_25m_heading_error"),
                "lookahead_30m_x": observation_named.get("lookahead_30m_x"),
                "lookahead_30m_y": observation_named.get("lookahead_30m_y"),
                "lookahead_30m_heading_error": observation_named.get("lookahead_30m_heading_error"),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": info.get("termination_reason"),
                "reward_terms_json": json.dumps(reward_terms, ensure_ascii=True, sort_keys=True),
            }
        )

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _write_global_map_plots(config_path: Path, stats_dir: Path, trace_csv_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("map_plot_skipped reason=matplotlib_not_installed", flush=True)
        return

    config = load_config(config_path)
    reference_path = ReferencePath.from_csv(config.path.csv_path)
    route_corridor = None
    if config.route.enabled:
        route_corridor = RouteCorridor.from_files(
            link_set_path=config.route.link_set_path,
            selection_path=config.route.corridor_selection_path,
            selection_key=config.route.corridor_selection_key,
            margin_m=config.route.corridor_margin_m,
        )
    rows = _read_trace_rows(trace_csv_path)
    if not rows:
        print("map_plot_skipped reason=no_trace_rows", flush=True)
        return
    _plot_trace_map(
        plt=plt,
        path=stats_dir / "global_trace_map.png",
        title="Evaluation Trajectories",
        reference_path=reference_path,
        route_corridor=route_corridor,
        rows=rows,
        failure_only=False,
    )
    _plot_trace_map(
        plt=plt,
        path=stats_dir / "failure_points_map.png",
        title="Failure Points",
        reference_path=reference_path,
        route_corridor=route_corridor,
        rows=rows,
        failure_only=True,
    )
    _plot_failure_heatmap(
        plt=plt,
        path=stats_dir / "failure_heatmap_map.png",
        title="Failure Point Heatmap",
        reference_path=reference_path,
        route_corridor=route_corridor,
        rows=rows,
    )


def _remove_existing_eval_outputs(stats_dir: Path, stats_prefix: str) -> None:
    for path in [
        stats_dir / f"{stats_prefix}.csv",
        stats_dir / f"{stats_prefix}.jsonl",
        stats_dir / "step_trace.csv",
    ]:
        if path.exists():
            path.unlink()


def _read_trace_rows(trace_csv_path: Path) -> list[dict]:
    with trace_csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_trace_map(
    *,
    plt,
    path: Path,
    title: str,
    reference_path: ReferencePath,
    route_corridor,
    rows: list[dict],
    failure_only: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    if route_corridor is not None:
        for _, ax0, ay0, bx0, by0, _ in route_corridor.segments:
            ax.plot([ax0, bx0], [ay0, by0], color="0.85", linewidth=1.0, zorder=1)
    path_x = [point.x for point in reference_path.points]
    path_y = [point.y for point in reference_path.points]
    ax.plot(path_x, path_y, color="black", linewidth=1.2, label="reference", zorder=2)

    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(str(row.get("episode_index")), []).append(row)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    for index, (episode, episode_rows) in enumerate(sorted(by_episode.items(), key=lambda item: int(item[0]))):
        xy = [
            (float(row["x"]), float(row["y"]))
            for row in episode_rows
            if row.get("x") not in {None, ""} and row.get("y") not in {None, ""}
        ]
        if not xy:
            continue
        color = color_cycle[index % len(color_cycle)]
        if not failure_only:
            xs, ys = zip(*xy)
            ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.75, zorder=3)
        end = episode_rows[-1]
        reason = str(end.get("termination_reason") or "")
        if failure_only and reason == "lap_completed":
            continue
        marker = "*" if reason == "lap_completed" else "x"
        ax.scatter(
            [float(end["x"])],
            [float(end["y"])],
            marker=marker,
            color=color,
            s=45,
            linewidths=1.6,
            zorder=4,
        )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.9", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved_map_plot={path}", flush=True)


def _plot_failure_heatmap(
    *,
    plt,
    path: Path,
    title: str,
    reference_path: ReferencePath,
    route_corridor,
    rows: list[dict],
) -> None:
    failures = []
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(str(row.get("episode_index")), []).append(row)
    for episode_rows in by_episode.values():
        end = episode_rows[-1]
        if str(end.get("termination_reason") or "") == "lap_completed":
            continue
        if end.get("x") in {None, ""} or end.get("y") in {None, ""}:
            continue
        failures.append((float(end["x"]), float(end["y"])))
    if not failures:
        print(f"map_plot_skipped path={path} reason=no_failures", flush=True)
        return
    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    if route_corridor is not None:
        for _, ax0, ay0, bx0, by0, _ in route_corridor.segments:
            ax.plot([ax0, bx0], [ay0, by0], color="0.88", linewidth=1.0, zorder=1)
    ax.plot(
        [point.x for point in reference_path.points],
        [point.y for point in reference_path.points],
        color="black",
        linewidth=1.2,
        zorder=2,
    )
    xs, ys = zip(*failures)
    heat = ax.hexbin(xs, ys, gridsize=45, cmap="inferno", mincnt=1, alpha=0.85, zorder=3)
    fig.colorbar(heat, ax=ax, label="failure count")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.9", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved_map_plot={path}", flush=True)


if __name__ == "__main__":
    main()
