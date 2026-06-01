from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None


def _strip_inline_comment(line: str) -> str:
    in_quote = False
    quote_char = ""
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if not in_quote:
                in_quote = True
                quote_char = char
            elif quote_char == char:
                in_quote = False
                quote_char = ""
            continue
        if char == "#" and not in_quote:
            return line[:index]
    return line


def _split_array_items(raw: str) -> list[str]:
    body = raw.strip()[1:-1].strip()
    if not body:
        return []
    items: list[str] = []
    start = 0
    in_quote = False
    quote_char = ""
    escaped = False
    for index, char in enumerate(body):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if not in_quote:
                in_quote = True
                quote_char = char
            elif quote_char == char:
                in_quote = False
                quote_char = ""
            continue
        if char == "," and not in_quote:
            items.append(body[start:index].strip())
            start = index + 1
    items.append(body[start:].strip())
    return items


def _parse_simple_toml_value(raw: str) -> Any:
    value = raw.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        return [_parse_simple_toml_value(item) for item in _split_array_items(value)]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return ast.literal_eval(value)
    try:
        if any(char in value for char in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_toml(path: Path) -> dict[str, Any]:
    # Minimal fallback for this project's flat TOML configs on Python 3.10 without tomli.
    raw: dict[str, Any] = {}
    current: dict[str, Any] = raw
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = _strip_inline_comment(line).strip()
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                section_name = stripped[1:-1].strip()
                current = raw.setdefault(section_name, {})
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            current[key.strip()] = _parse_simple_toml_value(value)
    return raw


def _load_toml(path: Path) -> dict[str, Any]:
    if tomllib is not None:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    return _load_simple_toml(path)


@dataclass
class RosConfig:
    enabled: bool = False
    node_name: str = "morai_rl"
    anonymous: bool = True
    user_id: str = "morai_rl"
    entity_id: str = "EGO"
    ego_topic: str = "/Ego_topic"
    imu_topic: str = "/imu"
    object_topic: str = "/Object_topic"
    sync_info_topic: str = "/SyncModeInfo"
    ctrl_topic: str = "/ctrl_cmd_0"
    sync_mode_cmd_service: str = "/SyncModeCmd"
    wait_for_tick_service: str = "/WaitForTick"
    sync_ctrl_cmd_service: str = "/SyncModeCtrlCmd"
    sync_set_gear_service: str = "/SyncModeSetGear"
    sync_scenario_load_service: str = "/SyncModeScenarioLoad"
    sync_scenario_load_post_ticks: int = 10
    sync_scenario_load_post_gear: int = 1
    use_sync_mode: bool = True
    start_sync_on_start: bool = True
    stop_sync_on_close: bool = False
    time_step: int = 20
    sensor_capture: bool = False
    service_timeout_sec: float = 5.0
    wait_for_tick_timeout_sec: float = 5.0


@dataclass
class EnvConfig:
    step_hz: float = 10.0
    action_repeat: int = 1
    max_steps: int = 1000
    target_speed_mps: float = 8.0
    action_mode: str = ""
    steering_only_control: bool = False
    steering_only_fixed_throttle: float = 0.0
    steering_only_fixed_brake: float = 0.0
    progress_reward_scale: float = 3.0
    alive_bonus: float = 0.0
    step_penalty: float = 0.02
    steering_delta_penalty_scale: float = 0.02
    brake_penalty_scale: float = 0.01
    boundary_proximity_penalty_scale: float = 0.2
    boundary_proximity_margin_m: float = 1.0
    lateral_error_penalty_scale: float = 0.0
    lateral_error_penalty_clip_m: float = 3.0
    heading_error_penalty_scale: float = 0.0
    heading_error_penalty_clip_rad: float = 1.0
    off_track_penalty: float = 100.0
    done_stop_duration_sec: float = 0.5
    off_track_distance_m: float = 3.0
    stop_speed_threshold_mps: float = 0.3
    stop_timeout_steps: int = 50
    state_timeout_sec: float = 1.0
    no_progress_window_steps: int = 20
    no_progress_epsilon_m: float = 0.5
    reverse_progress_window_steps: int = 10
    reverse_progress_threshold_m: float = -0.5
    stalled_penalty: float = 5.0
    blocked_collision_window_steps: int = 5
    blocked_collision_speed_threshold_mps: float = 0.3
    blocked_collision_min_throttle: float = 0.3
    blocked_collision_max_brake: float = 0.1
    blocked_collision_progress_epsilon_m: float = 0.15
    blocked_collision_boundary_overlap_ratio: float = 0.05
    blocked_collision_outside_ratio: float = 0.15
    blocked_collision_penalty: float = 100.0
    state_timeout_penalty: float = 20.0


@dataclass
class ResetConfig:
    command: str = ""
    scenario_load_enabled: bool = False
    scenario_load_file_name: str = ""
    scenario_load_file_names: list[str] = field(default_factory=list)
    scenario_selection_mode: str = "fixed"
    scenario_delete_all: bool = True
    scenario_load_network_connection_data: bool = True
    scenario_load_ego_vehicle_data: bool = True
    scenario_load_surrounding_vehicle_data: bool = True
    scenario_load_pedestrian_data: bool = True
    scenario_load_object_data: bool = True
    scenario_set_pause: bool = False
    reset_mode: str = "full_scenario_load"
    full_reload_interval: int = 0
    command_timeout_sec: float = 20.0
    min_reset_interval_sec: float = 8.0
    post_command_wait_sec: float = 2.0
    stable_speed_tolerance_mps: float = 0.3
    stable_position_tolerance_m: float = 0.5
    stable_frames_required: int = 5
    reset_timeout_sec: float = 25.0
    allow_unstable_reset: bool = True
    max_reset_attempts: int = 3
    reset_retry_wait_sec: float = 1.0


@dataclass
class PathConfig:
    csv_path: str = "../output/reference_path_centerline.csv"


@dataclass
class RouteConfig:
    enabled: bool = False
    link_set_path: str = ""
    corridor_selection_path: str = ""
    corridor_selection_key: str = "selected_link_ids"
    corridor_margin_m: float = 0.5


@dataclass
class ObservationConfig:
    mode: str = "vector"
    vector_profile: str = "legacy"
    lookahead_distances_m: list[float] = field(default_factory=lambda: [5.0, 10.0])
    guide_dropout_prob: float = 0.0


@dataclass
class BevConfig:
    width_px: int = 96
    height_px: int = 96
    front_range_m: float = 30.0
    rear_range_m: float = 10.0
    left_range_m: float = 15.0
    right_range_m: float = 15.0
    include_lane_marking: bool = False
    static_bev_npz_path: str = ""
    static_bev_metadata_path: str = ""
    lane_marking_path: str = ""
    corridor_boundary_width_m: float = 0.3
    centerline_width_m: float = 0.3
    lane_marking_min_width_m: float = 0.15
    ego_vehicle_length_m: float = 4.845
    ego_vehicle_width_m: float = 1.835
    ego_vehicle_offset_forward_m: float = 0.0


@dataclass
class RecoveryConfig:
    enabled: bool = False
    simulator_process_names: list[str] = field(default_factory=lambda: ["Simulator.x86_64"])
    terminate_timeout_sec: float = 5.0
    relaunch_command: str = ""
    relaunch_wait_sec: float = 20.0


@dataclass
class AppConfig:
    ros: RosConfig
    env: EnvConfig
    reset: ResetConfig
    path: PathConfig
    route: RouteConfig
    observation: ObservationConfig
    bev: BevConfig
    recovery: RecoveryConfig


def _merge_dataclass(dc_cls: type[Any], raw: dict[str, Any]) -> Any:
    defaults = dc_cls()
    merged = defaults.__dict__.copy()
    merged.update(raw)
    return dc_cls(**merged)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = _load_toml(config_path)

    ros = _merge_dataclass(RosConfig, raw.get("ros", {}))
    env = _merge_dataclass(EnvConfig, raw.get("env", {}))
    reset = _merge_dataclass(ResetConfig, raw.get("reset", {}))
    path_cfg = _merge_dataclass(PathConfig, raw.get("path", {}))
    route_cfg = _merge_dataclass(RouteConfig, raw.get("route", {}))
    observation_cfg = _merge_dataclass(ObservationConfig, raw.get("observation", {}))
    bev_cfg = _merge_dataclass(BevConfig, raw.get("bev", {}))
    recovery_cfg = _merge_dataclass(RecoveryConfig, raw.get("recovery", {}))

    if not Path(path_cfg.csv_path).is_absolute():
        path_cfg.csv_path = str((config_path.parent / path_cfg.csv_path).resolve())
    if route_cfg.link_set_path and not Path(route_cfg.link_set_path).is_absolute():
        route_cfg.link_set_path = str((config_path.parent / route_cfg.link_set_path).resolve())
    if route_cfg.corridor_selection_path and not Path(route_cfg.corridor_selection_path).is_absolute():
        route_cfg.corridor_selection_path = str((config_path.parent / route_cfg.corridor_selection_path).resolve())
    if bev_cfg.static_bev_npz_path and not Path(bev_cfg.static_bev_npz_path).is_absolute():
        bev_cfg.static_bev_npz_path = str((config_path.parent / bev_cfg.static_bev_npz_path).resolve())
    if bev_cfg.static_bev_metadata_path and not Path(bev_cfg.static_bev_metadata_path).is_absolute():
        bev_cfg.static_bev_metadata_path = str((config_path.parent / bev_cfg.static_bev_metadata_path).resolve())
    if bev_cfg.lane_marking_path and not Path(bev_cfg.lane_marking_path).is_absolute():
        bev_cfg.lane_marking_path = str((config_path.parent / bev_cfg.lane_marking_path).resolve())

    return AppConfig(
        ros=ros,
        env=env,
        reset=reset,
        path=path_cfg,
        route=route_cfg,
        observation=observation_cfg,
        bev=bev_cfg,
        recovery=recovery_cfg,
    )
