from __future__ import annotations

from dataclasses import dataclass
import math
import random
import shlex
import subprocess
import time

from morai_rl.core.types import VehicleState


@dataclass
class ResetOutcome:
    initial_state: VehicleState
    stable_frames: int
    scenario_name: str | None = None
    reset_strategy: str = "scenario_load"


class ScenarioResetManager:
    """
    Runs an external reset command and waits until the vehicle state is stable.

    The actual MORAI Scenario Load call depends on the user's integration method,
    so this class keeps the reset trigger as a shell command.
    """

    def __init__(
        self,
        vehicle_receiver,
        command: str = "",
        scenario_loader=None,
        multi_ego_client=None,
        reset_mode: str = "full_scenario_load",
        full_reload_interval: int = 0,
        scenario_file_names: list[str] | None = None,
        scenario_selection_mode: str = "fixed",
        command_timeout_sec: float = 20.0,
        min_reset_interval_sec: float = 0.0,
        post_command_wait_sec: float = 2.0,
        multi_ego_post_command_wait_sec: float = 1.5,
        multi_ego_position_tolerance_m: float = 1.0,
        multi_ego_yaw_tolerance_deg: float = 20.0,
        multi_ego_use_fixed_target: bool = False,
        multi_ego_target_x: float | None = None,
        multi_ego_target_y: float | None = None,
        multi_ego_target_z: float | None = None,
        multi_ego_target_roll_deg: float = 0.0,
        multi_ego_target_pitch_deg: float = 0.0,
        multi_ego_target_yaw_deg: float = 0.0,
        multi_ego_target_speed_kph: float = 0.0,
        stable_speed_tolerance_mps: float = 0.3,
        stable_position_tolerance_m: float = 0.5,
        stable_frames_required: int = 5,
        allow_unstable_reset: bool = True,
    ) -> None:
        self.vehicle_receiver = vehicle_receiver
        self.command = command.strip()
        self.scenario_loader = scenario_loader
        self.multi_ego_client = multi_ego_client
        self.reset_mode = reset_mode.strip().lower() or "full_scenario_load"
        self.full_reload_interval = max(0, int(full_reload_interval))
        self.scenario_file_names = [
            str(name).strip() for name in (scenario_file_names or []) if str(name).strip()
        ]
        self.scenario_selection_mode = scenario_selection_mode.strip().lower() or "fixed"
        self.command_timeout_sec = command_timeout_sec
        self.min_reset_interval_sec = max(0.0, float(min_reset_interval_sec))
        self.post_command_wait_sec = post_command_wait_sec
        self.multi_ego_post_command_wait_sec = max(0.0, float(multi_ego_post_command_wait_sec))
        self.multi_ego_position_tolerance_m = max(0.0, float(multi_ego_position_tolerance_m))
        self.multi_ego_yaw_tolerance_deg = max(0.0, float(multi_ego_yaw_tolerance_deg))
        self.multi_ego_use_fixed_target = bool(multi_ego_use_fixed_target)
        self.multi_ego_target_x = (
            None if multi_ego_target_x is None else float(multi_ego_target_x)
        )
        self.multi_ego_target_y = (
            None if multi_ego_target_y is None else float(multi_ego_target_y)
        )
        self.multi_ego_target_z = (
            None if multi_ego_target_z is None else float(multi_ego_target_z)
        )
        self.multi_ego_target_roll_deg = float(multi_ego_target_roll_deg)
        self.multi_ego_target_pitch_deg = float(multi_ego_target_pitch_deg)
        self.multi_ego_target_yaw_deg = float(multi_ego_target_yaw_deg)
        self.multi_ego_target_speed_mps = float(multi_ego_target_speed_kph) / 3.6
        self.stable_speed_tolerance_mps = stable_speed_tolerance_mps
        self.stable_position_tolerance_m = stable_position_tolerance_m
        self.stable_frames_required = stable_frames_required
        self.allow_unstable_reset = allow_unstable_reset
        self._scenario_index = -1
        self._last_reset_trigger_monotonic: float | None = None
        self._cached_reset_states: dict[str, VehicleState] = {}
        self._soft_reset_counts: dict[str, int] = {}

    def reset(self, timeout_sec: float) -> ResetOutcome:
        self.vehicle_receiver.clear_latest()
        self.vehicle_receiver.drain_socket()
        scenario_name = self._select_scenario_name()
        reset_strategy = self._select_reset_strategy(scenario_name)
        expected_state = None

        if reset_strategy == "multi_ego_setting":
            expected_state = self._get_multi_ego_target_state(scenario_name)
            self._run_multi_ego_reset(expected_state)
            time.sleep(self.multi_ego_post_command_wait_sec)
        else:
            scenario_name = self._run_full_reset_command(selected_scenario_name=scenario_name)
            time.sleep(self.post_command_wait_sec)

        self.vehicle_receiver.drain_socket()
        outcome = self._wait_until_stable(
            timeout_sec=timeout_sec,
            scenario_name=scenario_name,
            reset_strategy=reset_strategy,
            expected_state=expected_state,
        )
        if scenario_name:
            if reset_strategy == "scenario_load":
                self._cached_reset_states[scenario_name] = outcome.initial_state
                self._soft_reset_counts[scenario_name] = 0
            else:
                self._soft_reset_counts[scenario_name] = self._soft_reset_counts.get(scenario_name, 0) + 1
        return outcome

    def _run_full_reset_command(self, selected_scenario_name: str | None = None) -> str | None:
        self._enforce_min_reset_interval()
        if self.scenario_loader is not None:
            scenario_name = selected_scenario_name or self._select_scenario_name()
            self.scenario_loader.send(file_name=scenario_name)
            self._last_reset_trigger_monotonic = time.monotonic()
            return scenario_name
        if not self.command:
            return None
        subprocess.run(
            shlex.split(self.command),
            check=True,
            timeout=self.command_timeout_sec,
        )
        self._last_reset_trigger_monotonic = time.monotonic()
        return None

    def _run_multi_ego_reset(self, target_state: VehicleState) -> None:
        self._enforce_min_reset_interval()
        if self.multi_ego_client is None:
            raise RuntimeError("multi ego reset requested without a multi ego client")
        self.multi_ego_client.send_state(target_state)
        self._last_reset_trigger_monotonic = time.monotonic()

    def _enforce_min_reset_interval(self) -> None:
        if self.min_reset_interval_sec <= 0.0 or self._last_reset_trigger_monotonic is None:
            return
        elapsed_sec = time.monotonic() - self._last_reset_trigger_monotonic
        remaining_sec = self.min_reset_interval_sec - elapsed_sec
        if remaining_sec > 0.0:
            time.sleep(remaining_sec)

    def _select_scenario_name(self) -> str | None:
        if self.scenario_file_names:
            if self.scenario_selection_mode == "random":
                return random.choice(self.scenario_file_names)
            if self.scenario_selection_mode == "round_robin":
                self._scenario_index = (self._scenario_index + 1) % len(self.scenario_file_names)
                return self.scenario_file_names[self._scenario_index]
            return self.scenario_file_names[0]
        if self.scenario_loader is not None:
            return self.scenario_loader.file_name
        return None

    def _select_reset_strategy(self, scenario_name: str | None) -> str:
        if self.reset_mode != "multi_ego_setting":
            return "scenario_load"
        if self.multi_ego_client is None:
            return "scenario_load"
        if self._has_fixed_multi_ego_target():
            return "multi_ego_setting"
        if not scenario_name or scenario_name not in self._cached_reset_states:
            return "scenario_load"
        soft_reset_count = self._soft_reset_counts.get(scenario_name, 0)
        if self.full_reload_interval > 0 and soft_reset_count >= self.full_reload_interval:
            return "scenario_load"
        return "multi_ego_setting"

    def _get_multi_ego_target_state(self, scenario_name: str | None) -> VehicleState:
        if self._has_fixed_multi_ego_target():
            return VehicleState(
                timestamp_sec=0.0,
                entity_id="",
                x=self.multi_ego_target_x,
                y=self.multi_ego_target_y,
                z=self.multi_ego_target_z,
                roll_deg=self.multi_ego_target_roll_deg,
                pitch_deg=self.multi_ego_target_pitch_deg,
                yaw_deg=self.multi_ego_target_yaw_deg,
                vx=self.multi_ego_target_speed_mps,
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
        if scenario_name is None or scenario_name not in self._cached_reset_states:
            raise RuntimeError("multi ego reset requested without a cached target state")
        return self._cached_reset_states[scenario_name]

    def _has_fixed_multi_ego_target(self) -> bool:
        return (
            self.multi_ego_use_fixed_target
            and self.multi_ego_target_x is not None
            and self.multi_ego_target_y is not None
            and self.multi_ego_target_z is not None
        )

    def _wait_until_stable(
        self,
        timeout_sec: float,
        scenario_name: str | None,
        reset_strategy: str,
        expected_state: VehicleState | None = None,
    ) -> ResetOutcome:
        deadline = time.monotonic() + timeout_sec
        anchor = None
        stable_frames = 0
        latest_state = None

        while time.monotonic() < deadline:
            latest_state = self.vehicle_receiver.get_latest()
            if latest_state is None:
                time.sleep(0.05)
                continue

            if expected_state is not None and not self._matches_expected_state(latest_state, expected_state):
                anchor = None
                stable_frames = 0
                time.sleep(0.05)
                continue

            if anchor is None:
                anchor = latest_state
                stable_frames = 1
                time.sleep(0.05)
                continue

            is_slow = latest_state.speed_mps <= self.stable_speed_tolerance_mps
            dx = latest_state.x - anchor.x
            dy = latest_state.y - anchor.y
            pos_delta = (dx * dx + dy * dy) ** 0.5

            if is_slow and pos_delta <= self.stable_position_tolerance_m:
                stable_frames += 1
            else:
                anchor = latest_state
                stable_frames = 1

            if stable_frames >= self.stable_frames_required:
                return ResetOutcome(
                    initial_state=latest_state,
                    stable_frames=stable_frames,
                    scenario_name=scenario_name,
                    reset_strategy=reset_strategy,
                )

            time.sleep(0.05)

        if latest_state is not None and self.allow_unstable_reset:
            return ResetOutcome(
                initial_state=latest_state,
                stable_frames=stable_frames,
                scenario_name=scenario_name,
                reset_strategy=reset_strategy,
            )

        scenario_text = f" for scenario '{scenario_name}'" if scenario_name else ""
        strategy_text = (
            "MultiEgoSetting"
            if reset_strategy == "multi_ego_setting"
            else "Scenario Load"
        )
        raise TimeoutError(
            f"reset timeout{scenario_text}: vehicle did not settle into a stable state after {strategy_text}"
        )

    def _matches_expected_state(self, state: VehicleState, expected_state: VehicleState) -> bool:
        dx = state.x - expected_state.x
        dy = state.y - expected_state.y
        dz = state.z - expected_state.z
        position_error_m = math.sqrt(dx * dx + dy * dy + dz * dz)
        yaw_error_deg = abs(self._wrap_angle_deg(state.yaw_deg - expected_state.yaw_deg))
        return (
            position_error_m <= self.multi_ego_position_tolerance_m
            and yaw_error_deg <= self.multi_ego_yaw_tolerance_deg
        )

    @staticmethod
    def _wrap_angle_deg(angle_deg: float) -> float:
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg
