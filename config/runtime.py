from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class UdpConfig:
    host: str = "127.0.0.1"
    control_port: int = 6666
    control_bind_host: str = ""
    control_bind_port: int = 0
    vehicle_status_port: int = 9093
    object_port: int = 9094
    collision_port: int = 9092
    control_mode: str = "double3"
    entity_id: str = "EGO"


@dataclass
class EnvConfig:
    step_hz: float = 10.0
    action_repeat: int = 1
    max_steps: int = 1000
    target_speed_mps: float = 8.0
    steering_only_control: bool = False
    steering_only_fixed_throttle: float = 0.0
    steering_only_fixed_brake: float = 0.0
    progress_reward_scale: float = 3.0
    alive_bonus: float = 0.0
    step_penalty: float = 0.02
    steering_delta_penalty_scale: float = 0.02
    brake_penalty_scale: float = 0.01
    boundary_proximity_penalty_scale: float = 0.2
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
    scenario_load_bind_host: str = "127.0.0.1"
    scenario_load_bind_port: int = 9103
    scenario_load_destination_host: str = "127.0.0.1"
    scenario_load_destination_port: int = 9104
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
    multi_ego_setting_enabled: bool = False
    multi_ego_setting_bind_host: str = "127.0.0.1"
    multi_ego_setting_bind_port: int = 7604
    multi_ego_setting_destination_host: str = "127.0.0.1"
    multi_ego_setting_destination_port: int = 7504
    multi_ego_setting_ego_index: int = 0
    multi_ego_setting_camera_index: int = 0
    multi_ego_setting_gear: int = 4
    multi_ego_setting_ctrl_mode: int = 2
    multi_ego_setting_send_repeats: int = 3
    multi_ego_setting_send_interval_sec: float = 0.05
    multi_ego_setting_post_command_wait_sec: float = 1.5
    multi_ego_setting_position_tolerance_m: float = 1.0
    multi_ego_setting_yaw_tolerance_deg: float = 20.0
    multi_ego_setting_use_fixed_target: bool = False
    multi_ego_setting_target_x: float | None = None
    multi_ego_setting_target_y: float | None = None
    multi_ego_setting_target_z: float | None = None
    multi_ego_setting_target_roll_deg: float = 0.0
    multi_ego_setting_target_pitch_deg: float = 0.0
    multi_ego_setting_target_yaw_deg: float = 0.0
    multi_ego_setting_target_speed_kph: float = 0.0
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
    csv_path: str = "morai_rl/data/reference_path_example.csv"


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
class AppConfig:
    udp: UdpConfig
    env: EnvConfig
    reset: ResetConfig
    path: PathConfig
    route: RouteConfig
    observation: ObservationConfig
    bev: BevConfig


def _merge_dataclass(dc_cls: type[Any], raw: dict[str, Any]) -> Any:
    defaults = dc_cls()
    merged = defaults.__dict__.copy()
    merged.update(raw)
    return dc_cls(**merged)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    udp = _merge_dataclass(UdpConfig, raw.get("udp", {}))
    env = _merge_dataclass(EnvConfig, raw.get("env", {}))
    reset = _merge_dataclass(ResetConfig, raw.get("reset", {}))
    path_cfg = _merge_dataclass(PathConfig, raw.get("path", {}))
    route_cfg = _merge_dataclass(RouteConfig, raw.get("route", {}))
    observation_cfg = _merge_dataclass(ObservationConfig, raw.get("observation", {}))
    bev_cfg = _merge_dataclass(BevConfig, raw.get("bev", {}))

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
        udp=udp,
        env=env,
        reset=reset,
        path=path_cfg,
        route=route_cfg,
        observation=observation_cfg,
        bev=bev_cfg,
    )
