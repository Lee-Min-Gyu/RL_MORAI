from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


def normalize_angle_rad(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


@dataclass
class ControlCommand:
    throttle: float
    brake: float
    steering: float
    ctrl_mode: int = 2
    gear: int = 4
    long_cmd_type: int = 1
    velocity_kph: float = 0.0
    acceleration_mps2: float = 0.0

    @classmethod
    def zero(cls) -> "ControlCommand":
        return cls(throttle=0.0, brake=0.0, steering=0.0)

    def clipped(self) -> "ControlCommand":
        return ControlCommand(
            throttle=max(0.0, min(1.0, float(self.throttle))),
            brake=max(0.0, min(1.0, float(self.brake))),
            steering=float(self.steering),
            ctrl_mode=int(self.ctrl_mode),
            gear=int(self.gear),
            long_cmd_type=int(self.long_cmd_type),
            velocity_kph=float(self.velocity_kph),
            acceleration_mps2=float(self.acceleration_mps2),
        )


@dataclass
class VehicleState:
    timestamp_sec: float
    entity_id: str
    x: float
    y: float
    z: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float
    wx: float
    wy: float
    wz: float
    throttle: float
    brake: float
    steer_angle: float
    length_m: float | None = None
    width_m: float | None = None

    @property
    def speed_mps(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)

    def to_dict(self) -> dict[str, float | str]:
        data = asdict(self)
        data["speed_mps"] = self.speed_mps
        return data


@dataclass
class ObjectState:
    entity_id: str
    object_type: int
    x: float
    y: float
    z: float
    yaw_deg: float
    length: float
    width: float
    height: float
    vx: float
    vy: float
    vz: float


@dataclass
class CollisionObjectState:
    object_type: int
    object_id: int
    x: float
    y: float
    z: float
    global_offset_x: float
    global_offset_y: float
    global_offset_z: float

    @property
    def object_type_name(self) -> str:
        names = {
            -1: "ego_vehicle",
            0: "pedestrian",
            1: "vehicle",
            2: "object",
        }
        return names.get(self.object_type, f"unknown({self.object_type})")

    @property
    def is_empty(self) -> bool:
        return (
            self.object_type == 0
            and self.object_id == 0
            and self.x == 0.0
            and self.y == 0.0
            and self.z == 0.0
            and self.global_offset_x == 0.0
            and self.global_offset_y == 0.0
            and self.global_offset_z == 0.0
        )

    def to_dict(self) -> dict[str, float | int | str | bool]:
        data = asdict(self)
        data["object_type_name"] = self.object_type_name
        data["is_empty"] = self.is_empty
        return data


@dataclass
class CollisionState:
    timestamp_sec: float
    seconds: int
    nanos: int
    collisions: list[CollisionObjectState]
    header_name: str | None = None
    packet_len: int | None = None
    payload_len: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_sec": self.timestamp_sec,
            "seconds": self.seconds,
            "nanos": self.nanos,
            "header_name": self.header_name,
            "packet_len": self.packet_len,
            "payload_len": self.payload_len,
            "collisions": [item.to_dict() for item in self.collisions],
        }


@dataclass
class PathProjection:
    nearest_index: int
    distance_m: float
    progress_m: float
    progress_ratio: float
    path_heading_rad: float
    heading_error_rad: float
    lateral_error_m: float
    lookahead_heading_error_5m: float
    lookahead_heading_error_10m: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class Observation:
    values: Any
    named: dict[str, float]
    vector_values: list[float]
    bev: Any | None = None
