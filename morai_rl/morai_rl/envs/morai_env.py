from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import numpy as np

from morai_rl.config.runtime import AppConfig, load_config
from morai_rl.core.reset_manager import ScenarioResetManager
from morai_rl.core.sync_manager import StepClock
from morai_rl.core.types import ControlCommand, Observation, VehicleState
from morai_rl.envs.observation import build_observation, resolve_vector_observation_keys
from morai_rl.envs.reward import compute_reward
from morai_rl.envs.termination import evaluate_termination
from morai_rl.io.ros2 import (
    Ros2ControlClient,
    Ros2MultiEgoSettingClient,
    Ros2ObjectStatusReceiver,
    Ros2ScenarioLoadClient,
    Ros2VehicleStatusReceiver,
)
from morai_rl.maps.local_bev import LocalBeVRenderer
from morai_rl.maps.reference_path import ReferencePath
from morai_rl.maps.route_corridor import RouteCorridor


class MoraiRLEnv:
    """
    Minimal Gym-like environment over MORAI ROS2 topics.

    This starter intentionally stays simple:
    - fixed-size state vector
    - external Scenario Load reset trigger
    - optional object receiver, unused in the first phase
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.control_client = Ros2ControlClient(
            topic=config.ros2.control_topic,
            node_name=config.ros2.node_name,
            qos_depth=config.ros2.qos_depth,
        )
        self.vehicle_receiver = Ros2VehicleStatusReceiver(
            topic=config.ros2.vehicle_status_topic,
            node_name=config.ros2.node_name,
            qos_depth=config.ros2.qos_depth,
            entity_id=config.ros2.entity_id,
        )
        self.object_receiver = Ros2ObjectStatusReceiver(
            topic=config.ros2.object_topic,
            node_name=config.ros2.node_name,
            qos_depth=config.ros2.qos_depth,
        )
        self.reference_path = ReferencePath.from_csv(config.path.csv_path)
        self.route_corridor = None
        if config.route.enabled:
            self.route_corridor = RouteCorridor.from_files(
                link_set_path=config.route.link_set_path,
                selection_path=config.route.corridor_selection_path,
                selection_key=config.route.corridor_selection_key,
                margin_m=config.route.corridor_margin_m,
            )
        self.local_bev_renderer = None
        if config.observation.mode.strip().lower() in {"bev", "hybrid"}:
            self.local_bev_renderer = LocalBeVRenderer(
                reference_path=self.reference_path,
                route_corridor=self.route_corridor,
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
        scenario_loader = None
        if config.reset.scenario_load_enabled:
            scenario_loader = Ros2ScenarioLoadClient(
                topic=config.ros2.scenario_load_topic,
                node_name=config.ros2.node_name,
                qos_depth=config.ros2.qos_depth,
                file_name=config.reset.scenario_load_file_name,
                delete_all=config.reset.scenario_delete_all,
                load_network_connection_data=config.reset.scenario_load_network_connection_data,
                load_ego_vehicle_data=config.reset.scenario_load_ego_vehicle_data,
                load_surrounding_vehicle_data=config.reset.scenario_load_surrounding_vehicle_data,
                load_pedestrian_data=config.reset.scenario_load_pedestrian_data,
                load_object_data=config.reset.scenario_load_object_data,
                set_pause=config.reset.scenario_set_pause,
            )
        multi_ego_client = None
        if config.reset.multi_ego_setting_enabled:
            multi_ego_client = Ros2MultiEgoSettingClient(
                topic=config.ros2.multi_ego_setting_topic,
                node_name=config.ros2.node_name,
                qos_depth=config.ros2.qos_depth,
                ego_index=config.reset.multi_ego_setting_ego_index,
                camera_index=config.reset.multi_ego_setting_camera_index,
                gear=config.reset.multi_ego_setting_gear,
                ctrl_mode=config.reset.multi_ego_setting_ctrl_mode,
                send_repeats=config.reset.multi_ego_setting_send_repeats,
                send_interval_sec=config.reset.multi_ego_setting_send_interval_sec,
            )
        self.reset_manager = ScenarioResetManager(
            vehicle_receiver=self.vehicle_receiver,
            command=config.reset.command,
            scenario_loader=scenario_loader,
            multi_ego_client=multi_ego_client,
            reset_mode=config.reset.reset_mode,
            full_reload_interval=config.reset.full_reload_interval,
            scenario_file_names=config.reset.scenario_load_file_names,
            scenario_selection_mode=config.reset.scenario_selection_mode,
            command_timeout_sec=config.reset.command_timeout_sec,
            min_reset_interval_sec=config.reset.min_reset_interval_sec,
            post_command_wait_sec=config.reset.post_command_wait_sec,
            multi_ego_post_command_wait_sec=config.reset.multi_ego_setting_post_command_wait_sec,
            multi_ego_position_tolerance_m=config.reset.multi_ego_setting_position_tolerance_m,
            multi_ego_yaw_tolerance_deg=config.reset.multi_ego_setting_yaw_tolerance_deg,
            multi_ego_use_fixed_target=config.reset.multi_ego_setting_use_fixed_target,
            multi_ego_target_x=config.reset.multi_ego_setting_target_x,
            multi_ego_target_y=config.reset.multi_ego_setting_target_y,
            multi_ego_target_z=config.reset.multi_ego_setting_target_z,
            multi_ego_target_roll_deg=config.reset.multi_ego_setting_target_roll_deg,
            multi_ego_target_pitch_deg=config.reset.multi_ego_setting_target_pitch_deg,
            multi_ego_target_yaw_deg=config.reset.multi_ego_setting_target_yaw_deg,
            multi_ego_target_speed_kph=config.reset.multi_ego_setting_target_speed_kph,
            stable_speed_tolerance_mps=config.reset.stable_speed_tolerance_mps,
            stable_position_tolerance_m=config.reset.stable_position_tolerance_m,
            stable_frames_required=config.reset.stable_frames_required,
            allow_unstable_reset=config.reset.allow_unstable_reset,
        )
        self.clock = StepClock(config.env.step_hz)
        self.previous_action = ControlCommand.zero()
        self.last_state: VehicleState | None = None
        self.last_nearest_index: int | None = None
        self.last_progress_m = 0.0
        self.episode_progress_m = 0.0
        self.last_progress_delta_m = 0.0
        self.recent_progress_deltas: deque[float] = deque(
            maxlen=max(
                config.env.no_progress_window_steps,
                config.env.reverse_progress_window_steps,
                config.env.blocked_collision_window_steps,
            )
        )
        self.reset_count = 0
        self.step_count = 0
        self.current_scenario_name: str | None = None
        self.episode_start_wall_time: float | None = None
        self.last_reset_wall_time: float | None = None
        self.last_episode_end_reason: str | None = None
        self.last_episode_end_steps = 0
        self.last_episode_end_progress_m = 0.0
        self.last_episode_duration_sec = 0.0
        self._receivers_started = False
        self._drive_engaged_for_episode = False
        self.vector_observation_keys = resolve_vector_observation_keys(
            config.observation.vector_profile,
            config.observation.lookahead_distances_m,
        )

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "MoraiRLEnv":
        return cls(load_config(config_path))

    def reset(self) -> tuple[object, dict]:
        self._ensure_receivers()
        now = time.monotonic()
        seconds_since_previous_reset = (
            None if self.last_reset_wall_time is None else now - self.last_reset_wall_time
        )
        failed_reset_errors: list[str] = []
        outcome = None
        for attempt_index in range(max(1, self.config.reset.max_reset_attempts)):
            self._send_zero_for(duration_sec=0.3)
            try:
                outcome = self.reset_manager.reset(timeout_sec=self.config.reset.reset_timeout_sec)
                if attempt_index > 0:
                    print(
                        "reset recovered after retry "
                        f"{attempt_index + 1}/{self.config.reset.max_reset_attempts} "
                        f"(scenario={outcome.scenario_name})"
                    )
                break
            except TimeoutError as exc:
                failed_reset_errors.append(str(exc))
                if attempt_index + 1 >= max(1, self.config.reset.max_reset_attempts):
                    raise RuntimeError(
                        "reset failed after "
                        f"{self.config.reset.max_reset_attempts} attempts: "
                        + " | ".join(failed_reset_errors)
                    ) from exc
                print(
                    "reset attempt failed, trying next scenario: "
                    f"{attempt_index + 1}/{self.config.reset.max_reset_attempts} "
                    f"error={exc}"
                )
                time.sleep(self.config.reset.reset_retry_wait_sec)

        if outcome is None:
            raise RuntimeError("reset failed without producing an outcome")
        self.clock.reset()
        self.previous_action = ControlCommand.zero()
        self.last_state = outcome.initial_state
        self.current_scenario_name = outcome.scenario_name
        projection = self.reference_path.project(outcome.initial_state)
        self.last_progress_m = projection.progress_m
        self.last_nearest_index = projection.nearest_index
        self.episode_progress_m = 0.0
        self.last_progress_delta_m = 0.0
        self.recent_progress_deltas.clear()
        self.step_count = 0
        self._drive_engaged_for_episode = False
        self.reset_count += 1
        self.episode_start_wall_time = time.monotonic()
        self.last_reset_wall_time = self.episode_start_wall_time
        corridor_projection = (
            self.route_corridor.project(outcome.initial_state) if self.route_corridor is not None else None
        )
        observation = build_observation(
            state=outcome.initial_state,
            projection=projection,
            corridor_projection=corridor_projection,
            previous_action=self.previous_action,
            target_speed_mps=self.config.env.target_speed_mps,
            episode_progress_m=self.episode_progress_m,
            progress_delta_m=self.last_progress_delta_m,
            observation_mode=self.config.observation.mode,
            bev_renderer=self.local_bev_renderer,
            vector_profile=self.config.observation.vector_profile,
            guide_dropout_prob=self.config.observation.guide_dropout_prob,
            lookahead_distances_m=self.config.observation.lookahead_distances_m,
            reference_path=self.reference_path,
        )
        info = {
            "state": outcome.initial_state.to_dict(),
            "projection": projection.to_dict(),
            "corridor": corridor_projection.to_dict() if corridor_projection is not None else None,
            "stable_frames": outcome.stable_frames,
            "scenario_name": outcome.scenario_name,
            "reset_strategy": outcome.reset_strategy,
            "episode_progress_m": self.episode_progress_m,
            "progress_delta_m": self.last_progress_delta_m,
            "reset_attempts": 1 + len(failed_reset_errors),
            "reset_failures": failed_reset_errors,
            "reset_count": self.reset_count,
            "seconds_since_previous_reset": seconds_since_previous_reset,
            "previous_episode_reason": self.last_episode_end_reason,
            "previous_episode_steps": self.last_episode_end_steps,
            "previous_episode_progress_m": self.last_episode_end_progress_m,
            "previous_episode_duration_sec": self.last_episode_duration_sec,
            "observation_named": observation.named,
            "observation_vector_keys": self.vector_observation_keys,
            "observation_vector": observation.vector_values,
            "observation_mode": self.config.observation.mode,
            "observation_bev_channels": list(self.local_bev_renderer.channel_names)
            if self.local_bev_renderer is not None
            else [],
            "observation_bev_shape": list(observation.bev.shape)
            if observation.bev is not None
            else None,
        }
        return observation.values, info

    def step(
        self,
        action: ControlCommand | tuple[float, float, float] | list[float],
    ) -> tuple[object, float, bool, bool, dict]:
        command = self._coerce_action(action)
        self._engage_drive_for_episode()
        last_timestamp = self.last_state.timestamp_sec if self.last_state is not None else None
        for _ in range(self.config.env.action_repeat):
            self.control_client.send(command)
            self.clock.sleep()

        state = self._wait_for_latest_state(min_timestamp_sec=last_timestamp)
        projection = self.reference_path.project(
            state,
            hint_index=self.last_nearest_index,
            search_window=500,
        )
        corridor_projection = self.route_corridor.project(state) if self.route_corridor is not None else None
        progress_delta = self._compute_progress_delta(projection.progress_m)
        max_reasonable_progress_delta = max(
            1.0,
            (self.config.env.target_speed_mps / self.config.env.step_hz) * 3.0,
        )
        if progress_delta < -max_reasonable_progress_delta or progress_delta > max_reasonable_progress_delta:
            progress_delta = 0.0
        self.recent_progress_deltas.append(progress_delta)
        self.episode_progress_m += progress_delta
        self.last_progress_delta_m = progress_delta
        projection_off_track = (
            corridor_projection is not None and not corridor_projection.inside
        ) or (
            corridor_projection is None and projection.distance_m > self.config.env.off_track_distance_m
        )
        self.step_count += 1

        observation = build_observation(
            state=state,
            projection=projection,
            corridor_projection=corridor_projection,
            previous_action=self.previous_action,
            target_speed_mps=self.config.env.target_speed_mps,
            episode_progress_m=self.episode_progress_m,
            progress_delta_m=self.last_progress_delta_m,
            observation_mode=self.config.observation.mode,
            bev_renderer=self.local_bev_renderer,
            vector_profile=self.config.observation.vector_profile,
            guide_dropout_prob=self.config.observation.guide_dropout_prob,
            lookahead_distances_m=self.config.observation.lookahead_distances_m,
            reference_path=self.reference_path,
        )
        bev_contact = self._compute_bev_contact_metrics(observation)
        footprint_off_track = bool(bev_contact["available"]) and int(bev_contact["outside_pixels"]) > 0
        boundary_overlap_off_track = (
            bool(bev_contact["available"]) and int(bev_contact["boundary_overlap_pixels"]) > 0
        )
        off_track = projection_off_track or footprint_off_track or boundary_overlap_off_track

        stalled = (
            len(self.recent_progress_deltas) >= self.config.env.no_progress_window_steps
            and sum(list(self.recent_progress_deltas)[-self.config.env.no_progress_window_steps :])
            <= self.config.env.no_progress_epsilon_m
        )
        blocked_collision = False
        reverse_progress = (
            len(self.recent_progress_deltas) >= self.config.env.reverse_progress_window_steps
            and sum(list(self.recent_progress_deltas)[-self.config.env.reverse_progress_window_steps :])
            <= self.config.env.reverse_progress_threshold_m
        )
        reward, reward_terms = compute_reward(
            progress_delta_m=progress_delta,
            projection=projection,
            corridor_projection=corridor_projection,
            action=command,
            previous_action=self.previous_action,
            off_track=off_track,
            blocked_collision=blocked_collision,
            stalled=stalled,
            progress_reward_scale=self.config.env.progress_reward_scale,
            alive_bonus=self.config.env.alive_bonus,
            step_penalty_value=self.config.env.step_penalty,
            steering_delta_penalty_scale=self.config.env.steering_delta_penalty_scale,
            brake_penalty_scale=self.config.env.brake_penalty_scale,
            lateral_error_penalty_scale=self.config.env.lateral_error_penalty_scale,
            lateral_error_penalty_clip_m=self.config.env.lateral_error_penalty_clip_m,
            heading_error_penalty_scale=self.config.env.heading_error_penalty_scale,
            heading_error_penalty_clip_rad=self.config.env.heading_error_penalty_clip_rad,
            boundary_proximity_penalty_scale=self.config.env.boundary_proximity_penalty_scale,
            boundary_proximity_margin_m=self.config.env.boundary_proximity_margin_m,
            off_track_penalty_value=self.config.env.off_track_penalty,
            stalled_penalty_value=self.config.env.stalled_penalty,
        )
        terminated, truncated, reason = evaluate_termination(
            step_count=self.step_count,
            max_steps=self.config.env.max_steps,
            off_track=off_track,
            blocked_collision=blocked_collision,
            reverse_progress=reverse_progress,
            stalled=stalled,
        )
        self.previous_action = command
        self.last_state = state
        self.last_nearest_index = projection.nearest_index
        self.last_progress_m = projection.progress_m

        info = {
            "state": state.to_dict(),
            "projection": projection.to_dict(),
            "corridor": corridor_projection.to_dict() if corridor_projection is not None else None,
            "reward_terms": reward_terms,
            "episode_progress_m": self.episode_progress_m,
            "progress_delta_m": self.last_progress_delta_m,
            "step_count": self.step_count,
            "episode_duration_sec": (
                0.0
                if self.episode_start_wall_time is None
                else max(0.0, time.monotonic() - self.episode_start_wall_time)
            ),
            "scenario_name": self.current_scenario_name,
            "observation_named": observation.named,
            "observation_vector_keys": self.vector_observation_keys,
            "observation_vector": observation.vector_values,
            "observation_mode": self.config.observation.mode,
            "observation_bev_channels": list(self.local_bev_renderer.channel_names)
            if self.local_bev_renderer is not None
            else [],
            "observation_bev_shape": list(observation.bev.shape)
            if observation.bev is not None
            else None,
            "bev_contact": bev_contact,
            "blocked_collision": blocked_collision,
            "termination_reason": reason,
        }
        if terminated or truncated:
            self.last_episode_end_reason = reason
            self.last_episode_end_steps = self.step_count
            self.last_episode_end_progress_m = self.episode_progress_m
            self.last_episode_duration_sec = (
                0.0
                if self.episode_start_wall_time is None
                else max(0.0, time.monotonic() - self.episode_start_wall_time)
            )
            self._send_zero_for(
                duration_sec=self.config.env.done_stop_duration_sec,
                allow_create=True,
            )
        return observation.values, reward, terminated, truncated, info

    def close(self) -> None:
        self._send_zero_for(duration_sec=0.1, allow_create=False)
        if self._receivers_started:
            self.vehicle_receiver.stop()
            self.object_receiver.stop()
            self._receivers_started = False
        if getattr(self.reset_manager, "scenario_loader", None) is not None:
            self.reset_manager.scenario_loader.close()
        if getattr(self.reset_manager, "multi_ego_client", None) is not None:
            self.reset_manager.multi_ego_client.close()
        self.control_client.close()

    def _ensure_receivers(self) -> None:
        if self._receivers_started:
            return
        self.vehicle_receiver.start()
        self._receivers_started = True

    def _wait_for_latest_state(self, min_timestamp_sec: float | None = None) -> VehicleState:
        return self.vehicle_receiver.wait_for_state(
            self.config.env.state_timeout_sec,
            min_timestamp_sec=min_timestamp_sec,
        )

    def _send_zero_for(self, duration_sec: float, allow_create: bool = True) -> None:
        if not allow_create and self.control_client.socket is None:
            return
        end_time = time.monotonic() + duration_sec
        while time.monotonic() < end_time:
            self.control_client.send(ControlCommand.zero())
            time.sleep(0.05)

    def _engage_drive_for_episode(self) -> None:
        if self._drive_engaged_for_episode:
            return
        self._drive_engaged_for_episode = True
        if not self.config.reset.multi_ego_setting_drive_on_first_step:
            return
        multi_ego_client = getattr(self.reset_manager, "multi_ego_client", None)
        if multi_ego_client is None or self.last_state is None:
            return
        multi_ego_client.send_state(
            self.last_state,
            gear=self.config.reset.multi_ego_setting_drive_gear,
            ctrl_mode=self.config.reset.multi_ego_setting_drive_ctrl_mode,
        )
        if self.config.reset.multi_ego_setting_drive_wait_sec > 0.0:
            time.sleep(self.config.reset.multi_ego_setting_drive_wait_sec)

    def timeout_transition(
        self,
        reason: str,
        error_message: str,
    ) -> tuple[object, float, bool, bool, dict]:
        if self.last_state is not None:
            projection = self.reference_path.project(
                self.last_state,
                hint_index=self.last_nearest_index,
                search_window=500,
            )
            corridor_projection = (
                self.route_corridor.project(self.last_state) if self.route_corridor is not None else None
            )
            observation = build_observation(
                state=self.last_state,
                projection=projection,
                corridor_projection=corridor_projection,
                previous_action=self.previous_action,
                target_speed_mps=self.config.env.target_speed_mps,
                episode_progress_m=self.episode_progress_m,
                progress_delta_m=0.0,
                observation_mode=self.config.observation.mode,
                bev_renderer=self.local_bev_renderer,
                vector_profile=self.config.observation.vector_profile,
                guide_dropout_prob=self.config.observation.guide_dropout_prob,
                lookahead_distances_m=self.config.observation.lookahead_distances_m,
                reference_path=self.reference_path,
            )
            state_dict = self.last_state.to_dict()
            projection_dict = projection.to_dict()
            corridor_dict = corridor_projection.to_dict() if corridor_projection is not None else None
        else:
            observation = build_observation(
                state=VehicleState(
                    timestamp_sec=0.0,
                    entity_id=self.config.ros2.entity_id,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    roll_deg=0.0,
                    pitch_deg=0.0,
                    yaw_deg=0.0,
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                    ax=0.0,
                    ay=0.0,
                    az=0.0,
                    wx=0.0,
                    wy=0.0,
                    wz=0.0,
                    throttle=0.0,
                    brake=0.0,
                    steer_angle=0.0,
                ),
                projection=self.reference_path.project(
                    VehicleState(
                        timestamp_sec=0.0,
                        entity_id=self.config.ros2.entity_id,
                        x=self.reference_path.points[0].x,
                        y=self.reference_path.points[0].y,
                        z=0.0,
                        roll_deg=0.0,
                        pitch_deg=0.0,
                        yaw_deg=0.0,
                        vx=0.0,
                        vy=0.0,
                        vz=0.0,
                        ax=0.0,
                        ay=0.0,
                        az=0.0,
                        wx=0.0,
                        wy=0.0,
                        wz=0.0,
                        throttle=0.0,
                        brake=0.0,
                        steer_angle=0.0,
                    )
                ),
                corridor_projection=None,
                previous_action=self.previous_action,
                target_speed_mps=self.config.env.target_speed_mps,
                episode_progress_m=self.episode_progress_m,
                progress_delta_m=0.0,
                observation_mode=self.config.observation.mode,
                bev_renderer=self.local_bev_renderer,
                vector_profile=self.config.observation.vector_profile,
                guide_dropout_prob=self.config.observation.guide_dropout_prob,
                lookahead_distances_m=self.config.observation.lookahead_distances_m,
                reference_path=self.reference_path,
            )
            state_dict = None
            projection_dict = None
            corridor_dict = None

        info = {
            "state": state_dict,
            "projection": projection_dict,
            "corridor": corridor_dict,
            "reward_terms": {
                "state_timeout_penalty": self.config.env.state_timeout_penalty,
            },
            "episode_progress_m": self.episode_progress_m,
            "progress_delta_m": 0.0,
            "step_count": self.step_count,
            "episode_duration_sec": (
                0.0
                if self.episode_start_wall_time is None
                else max(0.0, time.monotonic() - self.episode_start_wall_time)
            ),
            "scenario_name": self.current_scenario_name,
            "observation_named": observation.named,
            "observation_vector_keys": self.vector_observation_keys,
            "observation_vector": observation.vector_values,
            "observation_mode": self.config.observation.mode,
            "observation_bev_channels": list(self.local_bev_renderer.channel_names)
            if self.local_bev_renderer is not None
            else [],
            "observation_bev_shape": list(observation.bev.shape)
            if observation.bev is not None
            else None,
            "termination_reason": reason,
            "env_error": error_message,
        }
        self.last_episode_end_reason = reason
        self.last_episode_end_steps = self.step_count
        self.last_episode_end_progress_m = self.episode_progress_m
        self.last_episode_duration_sec = (
            0.0
            if self.episode_start_wall_time is None
            else max(0.0, time.monotonic() - self.episode_start_wall_time)
        )
        self._send_zero_for(
            duration_sec=self.config.env.done_stop_duration_sec,
            allow_create=True,
        )
        return (
            observation.values,
            -self.config.env.state_timeout_penalty,
            True,
            False,
            info,
        )

    def _compute_progress_delta(self, current_progress_m: float) -> float:
        progress_delta = current_progress_m - self.last_progress_m
        total_length_m = self.reference_path.total_length_m
        if total_length_m > 0.0:
            half_length_m = total_length_m * 0.5
            if progress_delta < -half_length_m:
                progress_delta += total_length_m
            elif progress_delta > half_length_m:
                progress_delta -= total_length_m
        return progress_delta

    def _compute_bev_contact_metrics(self, observation: Observation) -> dict[str, float | int | bool]:
        metrics: dict[str, float | int | bool] = {
            "available": False,
            "ego_pixels": 0,
            "boundary_overlap_pixels": 0,
            "outside_pixels": 0,
            "boundary_overlap_ratio": 0.0,
            "outside_ratio": 0.0,
        }
        if observation.bev is None or self.local_bev_renderer is None:
            return metrics

        channel_indices = {
            name: index for index, name in enumerate(self.local_bev_renderer.channel_names)
        }
        area_index = channel_indices.get("corridor_area")
        boundary_index = channel_indices.get("corridor_boundary")
        ego_index = channel_indices.get("ego_footprint")
        if area_index is None or boundary_index is None or ego_index is None:
            return metrics

        ego_mask = observation.bev[ego_index] > 0
        ego_pixels = int(np.count_nonzero(ego_mask))
        if ego_pixels <= 0:
            return metrics

        boundary_overlap_pixels = int(
            np.count_nonzero((observation.bev[boundary_index] > 0) & ego_mask)
        )
        outside_pixels = int(
            np.count_nonzero((observation.bev[area_index] <= 0) & ego_mask)
        )
        metrics.update(
            {
                "available": True,
                "ego_pixels": ego_pixels,
                "boundary_overlap_pixels": boundary_overlap_pixels,
                "outside_pixels": outside_pixels,
                "boundary_overlap_ratio": boundary_overlap_pixels / float(ego_pixels),
                "outside_ratio": outside_pixels / float(ego_pixels),
            }
        )
        return metrics

    def _detect_blocked_collision(
        self,
        state: VehicleState,
        action: ControlCommand,
        off_track: bool,
        bev_contact: dict[str, float | int | bool],
    ) -> bool:
        if off_track:
            return False
        window_steps = max(1, int(self.config.env.blocked_collision_window_steps))
        if len(self.recent_progress_deltas) < window_steps:
            return False

        recent_progress = sum(list(self.recent_progress_deltas)[-window_steps:])
        trying_to_move = (
            action.throttle >= self.config.env.blocked_collision_min_throttle
            and action.brake <= self.config.env.blocked_collision_max_brake
        )
        low_speed = state.speed_mps <= self.config.env.blocked_collision_speed_threshold_mps
        low_progress = recent_progress <= self.config.env.blocked_collision_progress_epsilon_m
        boundary_overlap_ratio = float(bev_contact["boundary_overlap_ratio"])
        outside_ratio = float(bev_contact["outside_ratio"])
        boundary_contact = (
            bool(bev_contact["available"])
            and (
                boundary_overlap_ratio >= self.config.env.blocked_collision_boundary_overlap_ratio
                or outside_ratio >= self.config.env.blocked_collision_outside_ratio
            )
        )
        return trying_to_move and low_speed and low_progress and boundary_contact

    @staticmethod
    def _coerce_action(action: ControlCommand | tuple[float, float, float] | list[float]) -> ControlCommand:
        if isinstance(action, ControlCommand):
            return action.clipped()
        if len(action) != 3:
            raise ValueError("action must be (throttle, brake, steering)")
        return ControlCommand(
            throttle=float(action[0]),
            brake=float(action[1]),
            steering=float(action[2]),
        ).clipped()
