from __future__ import annotations

from dataclasses import dataclass
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
    Runs a ROS sync scenario load or an external reset command, then waits
    until the vehicle state is stable.
    """

    def __init__(
        self,
        vehicle_receiver,
        command: str = "",
        scenario_loader=None,
        reset_mode: str = "full_scenario_load",
        full_reload_interval: int = 0,
        scenario_file_names: list[str] | None = None,
        scenario_selection_mode: str = "fixed",
        command_timeout_sec: float = 20.0,
        min_reset_interval_sec: float = 0.0,
        post_command_wait_sec: float = 2.0,
        stable_speed_tolerance_mps: float = 0.3,
        stable_position_tolerance_m: float = 0.5,
        stable_frames_required: int = 5,
        allow_unstable_reset: bool = True,
    ) -> None:
        self.vehicle_receiver = vehicle_receiver
        self.command = command.strip()
        self.scenario_loader = scenario_loader
        self.reset_mode = reset_mode.strip().lower() or "full_scenario_load"
        self.full_reload_interval = max(0, int(full_reload_interval))
        self.scenario_file_names = [
            str(name).strip() for name in (scenario_file_names or []) if str(name).strip()
        ]
        self.scenario_selection_mode = scenario_selection_mode.strip().lower() or "fixed"
        self.command_timeout_sec = command_timeout_sec
        self.min_reset_interval_sec = max(0.0, float(min_reset_interval_sec))
        self.post_command_wait_sec = post_command_wait_sec
        self.stable_speed_tolerance_mps = stable_speed_tolerance_mps
        self.stable_position_tolerance_m = stable_position_tolerance_m
        self.stable_frames_required = stable_frames_required
        self.allow_unstable_reset = allow_unstable_reset
        self._scenario_index = -1
        self._last_reset_trigger_monotonic: float | None = None

    def reset(self, timeout_sec: float) -> ResetOutcome:
        self.vehicle_receiver.clear_latest()
        self.vehicle_receiver.drain_socket()
        scenario_name = self._select_scenario_name()
        scenario_name = self._run_full_reset_command(selected_scenario_name=scenario_name)
        time.sleep(self.post_command_wait_sec)

        self.vehicle_receiver.drain_socket()
        return self._wait_until_stable(
            timeout_sec=timeout_sec,
            scenario_name=scenario_name,
            reset_strategy="scenario_load",
        )

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

    def _wait_until_stable(
        self,
        timeout_sec: float,
        scenario_name: str | None,
        reset_strategy: str,
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
        raise TimeoutError(
            f"reset timeout{scenario_text}: vehicle did not settle into a stable state after Scenario Load"
        )
