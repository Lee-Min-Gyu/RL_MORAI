from __future__ import annotations

from dataclasses import dataclass

from morai_rl.core.types import ControlCommand


@dataclass
class SimpleDriverConfig:
    steer_sign: float = 1.0
    k_heading: float = 0.9
    k_lateral: float = 0.35
    k_speed: float = 0.25
    k_brake: float = 0.35
    max_steering: float = 0.6


class SimpleLaneFollower:
    """
    Lightweight rule-based driver used to validate reset/step/state first.

    If steering turns the wrong direction in MORAI, flip steer_sign.
    """

    def __init__(self, config: SimpleDriverConfig | None = None) -> None:
        self.config = config or SimpleDriverConfig()

    def act(self, observation_named: dict[str, float]) -> ControlCommand:
        heading_error = observation_named["heading_error_rad"]
        lateral_error = observation_named["lateral_error_m"]
        speed = observation_named["speed_mps"]
        target_speed = observation_named["target_speed_mps"]

        raw_steering = self.config.steer_sign * (
            self.config.k_heading * heading_error + self.config.k_lateral * lateral_error
        )
        steering = max(-self.config.max_steering, min(self.config.max_steering, raw_steering))

        speed_error = target_speed - speed
        if speed_error >= 0.0:
            throttle = min(1.0, self.config.k_speed * speed_error)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, self.config.k_brake * (-speed_error))

        return ControlCommand(throttle=throttle, brake=brake, steering=steering)
