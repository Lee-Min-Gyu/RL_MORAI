from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    from stable_baselines3 import PPO
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    PPO = None
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None

from morai_rl.config.runtime import load_config
from morai_rl.core.types import ControlCommand
from morai_rl.envs.observation import build_observation
from morai_rl.io.ros_sync import RosControlClient, RosVehicleStatusReceiver
from morai_rl.maps.local_bev import LocalBeVRenderer
from morai_rl.maps.reference_path import ReferencePath
from morai_rl.maps.route_corridor import RouteCorridor

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "stage1_ros_sync_config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained PPO policy in real-time ROS async mode without env reset/reward."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", required=True, help="Path to a PPO .zip checkpoint or path without .zip suffix.")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 means run until Ctrl-C.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no step limit.")
    parser.add_argument("--state-timeout-sec", type=float, default=1.0)
    parser.add_argument("--search-window", type=int, default=200)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Compute actions but do not publish control commands.")
    parser.add_argument(
        "--release-sync-mode",
        action="store_true",
        help="Try to release MORAI sync mode before publishing normal CtrlCmd messages.",
    )
    parser.add_argument("--no-zero-on-exit", dest="zero_on_exit", action="store_false", default=True)
    return parser.parse_args()


def main() -> None:
    if PPO is None:
        raise ModuleNotFoundError(
            "stable-baselines3 is required. Install it with `pip install stable-baselines3`."
        ) from _SB3_IMPORT_ERROR

    args = parse_args()
    config = load_config(args.config)
    model_path = _resolve_model_path(args.model)
    model = PPO.load(str(model_path), device="auto")
    reference_path = ReferencePath.from_csv(config.path.csv_path)
    if config.path.width_csv_path:
        reference_path.attach_widths_from_csv(config.path.width_csv_path)
    route_corridor = _build_route_corridor(config)
    bev_renderer = _build_bev_renderer(config, reference_path, route_corridor)

    receiver = RosVehicleStatusReceiver(
        topic=config.ros.ego_topic,
        entity_id=config.ros.entity_id,
        imu_topic=config.ros.imu_topic,
        node_name=config.ros.node_name,
        anonymous=config.ros.anonymous,
    )
    controller = RosControlClient(
        ctrl_topic=config.ros.ctrl_topic,
        use_sync_mode=False,
        user_id=config.ros.user_id,
        time_step=config.ros.time_step,
        sensor_capture=False,
        node_name=config.ros.node_name,
        anonymous=config.ros.anonymous,
    )

    if args.release_sync_mode:
        _try_release_sync_mode(config)

    print(f"async_drive_model={model_path}", flush=True)
    print(
        "async_drive_start "
        f"ctrl_topic={config.ros.ctrl_topic} ego_topic={config.ros.ego_topic} "
        f"hz={float(args.hz):.2f} deterministic={bool(args.deterministic)} "
        f"dry_run={bool(args.dry_run)}",
        flush=True,
    )

    previous_action = ControlCommand.zero()
    last_nearest_index = None
    last_progress_m = 0.0
    first_progress_m = None
    step_count = 0
    started_at = time.monotonic()
    period_sec = 1.0 / max(0.1, float(args.hz))

    receiver.start()
    try:
        while True:
            loop_started = time.monotonic()
            if args.duration_sec > 0.0 and loop_started - started_at >= float(args.duration_sec):
                break
            if args.max_steps > 0 and step_count >= int(args.max_steps):
                break

            state = receiver.wait_for_state(timeout_sec=float(args.state_timeout_sec))
            projection = reference_path.project(
                state,
                hint_index=last_nearest_index,
                search_window=max(1, int(args.search_window)),
            )
            corridor_projection = route_corridor.project(state) if route_corridor is not None else None
            progress_delta_m = projection.progress_m - last_progress_m if step_count > 0 else 0.0
            if first_progress_m is None:
                first_progress_m = projection.progress_m
            episode_progress_m = projection.progress_m - first_progress_m
            observation = build_observation(
                state=state,
                projection=projection,
                corridor_projection=corridor_projection,
                previous_action=previous_action,
                target_speed_mps=config.env.target_speed_mps,
                episode_progress_m=episode_progress_m,
                progress_delta_m=progress_delta_m,
                observation_mode=config.observation.mode,
                bev_renderer=bev_renderer,
                vector_profile=config.observation.vector_profile,
                guide_dropout_prob=0.0,
                lookahead_distances_m=config.observation.lookahead_distances_m,
                reference_path=reference_path,
                ego_vehicle_width_m=config.bev.ego_vehicle_width_m,
            )
            action, _ = model.predict(observation.values, deterministic=bool(args.deterministic))
            command = _action_to_command(np.asarray(action, dtype=np.float32), config.env).clipped()
            if not args.dry_run:
                controller.send(command)
            previous_action = command
            last_nearest_index = projection.nearest_index
            last_progress_m = projection.progress_m
            step_count += 1

            if args.log_every > 0 and (step_count == 1 or step_count % int(args.log_every) == 0):
                print(
                    "async_drive_step "
                    f"step={step_count} progress_m={episode_progress_m:+.2f} "
                    f"speed={state.speed_mps:.2f} "
                    f"lat={projection.lateral_error_m:+.3f} "
                    f"head={projection.heading_error_rad:+.3f} "
                    f"action0={float(np.ravel(action)[0]):+.3f} "
                    f"action1={float(np.ravel(action)[1]):+.3f} "
                    f"throttle={command.throttle:.3f} brake={command.brake:.3f} "
                    f"steering={command.steering:+.3f}",
                    flush=True,
                )

            elapsed = time.monotonic() - loop_started
            if elapsed < period_sec:
                time.sleep(period_sec - elapsed)
    except KeyboardInterrupt:
        print("async_drive_interrupted", flush=True)
    finally:
        if args.zero_on_exit and not args.dry_run:
            try:
                controller.send(ControlCommand.zero())
                print("async_drive_sent_zero_on_exit", flush=True)
            except Exception as exc:
                print(f"async_drive_zero_on_exit_failed error={exc}", flush=True)
        receiver.close()
        controller.close()


def _resolve_model_path(model: str) -> Path:
    path = Path(model)
    if path.is_file():
        return path
    if path.suffix != ".zip" and path.with_suffix(".zip").is_file():
        return path.with_suffix(".zip")
    raise FileNotFoundError(f"model not found: {model}")


def _build_route_corridor(config):
    if not config.route.enabled:
        return None
    return RouteCorridor.from_files(
        link_set_path=config.route.link_set_path,
        selection_path=config.route.corridor_selection_path,
        selection_key=config.route.corridor_selection_key,
        margin_m=config.route.corridor_margin_m,
    )


def _build_bev_renderer(config, reference_path: ReferencePath, route_corridor):
    if config.observation.mode.strip().lower() not in {"bev", "hybrid"}:
        return None
    return LocalBeVRenderer(
        reference_path=reference_path,
        route_corridor=route_corridor,
        link_set_path=config.route.link_set_path,
        lane_marking_path=config.bev.lane_marking_path,
        width_px=config.bev.width_px,
        height_px=config.bev.height_px,
        front_range_m=config.bev.front_range_m,
        rear_range_m=config.bev.rear_range_m,
        left_range_m=config.bev.left_range_m,
        right_range_m=config.bev.right_range_m,
        include_lane_marking=config.bev.include_lane_marking,
        static_bev_npz_path=config.bev.static_bev_npz_path,
        static_bev_metadata_path=config.bev.static_bev_metadata_path,
        corridor_boundary_width_m=config.bev.corridor_boundary_width_m,
        centerline_width_m=config.bev.centerline_width_m,
        lane_marking_min_width_m=config.bev.lane_marking_min_width_m,
        ego_vehicle_length_m=config.bev.ego_vehicle_length_m,
        ego_vehicle_width_m=config.bev.ego_vehicle_width_m,
        ego_vehicle_offset_forward_m=config.bev.ego_vehicle_offset_forward_m,
    )


def _action_to_command(action: np.ndarray, env_config) -> ControlCommand:
    action_np = np.ravel(action).astype(np.float32)
    mode = env_config.action_mode.strip().lower()
    if not mode:
        mode = "full"
    if mode == "steering_only":
        steering = float(action_np[1] if action_np.shape[0] > 1 else action_np[0])
        return ControlCommand(
            throttle=float(env_config.steering_only_fixed_throttle),
            brake=float(env_config.steering_only_fixed_brake),
            steering=steering,
        )
    if mode == "throttle_brake_steering":
        throttle_brake = float(action_np[0])
        return ControlCommand(
            throttle=max(0.0, throttle_brake),
            brake=max(0.0, -throttle_brake),
            steering=float(action_np[1]),
        )
    if mode == "throttle_steering":
        return ControlCommand(
            throttle=max(0.0, float(action_np[0])),
            brake=float(env_config.steering_only_fixed_brake),
            steering=float(action_np[1]),
        )
    if mode == "full":
        if action_np.shape[0] == 2:
            throttle_brake = float(action_np[0])
            return ControlCommand(
                throttle=max(0.0, throttle_brake),
                brake=max(0.0, -throttle_brake),
                steering=float(action_np[1]),
            )
        return ControlCommand(
            throttle=float(action_np[0]),
            brake=float(action_np[1]),
            steering=float(action_np[2]),
        )
    if mode == "full_3d":
        return ControlCommand(
            throttle=float(action_np[0]),
            brake=float(action_np[1]),
            steering=float(action_np[2]),
        )
    raise ValueError(f"unsupported action_mode for async driving: {env_config.action_mode!r}")


def _try_release_sync_mode(config) -> None:
    releaser = RosControlClient(
        ctrl_topic=config.ros.ctrl_topic,
        use_sync_mode=True,
        user_id=config.ros.user_id,
        time_step=config.ros.time_step,
        sensor_capture=False,
        sync_info_topic=config.ros.sync_info_topic,
        sync_mode_cmd_service=config.ros.sync_mode_cmd_service,
        sync_ctrl_cmd_service=config.ros.sync_ctrl_cmd_service,
        sync_set_gear_service=config.ros.sync_set_gear_service,
        wait_for_tick_service=config.ros.wait_for_tick_service,
        service_timeout_sec=config.ros.service_timeout_sec,
        wait_for_tick_timeout_sec=config.ros.wait_for_tick_timeout_sec,
        start_sync_on_start=False,
        stop_sync_on_close=False,
        node_name=config.ros.node_name,
        anonymous=config.ros.anonymous,
    )
    try:
        releaser._ensure_started()
        released = releaser.release_sync_mode()
        print(f"release_sync_mode result={released}", flush=True)
    finally:
        releaser.close()


if __name__ == "__main__":
    main()
