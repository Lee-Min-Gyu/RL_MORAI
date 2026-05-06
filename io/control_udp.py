from __future__ import annotations

import json
import socket
import struct

from morai_rl.core.types import ControlCommand


def _pad_id_24(value: str) -> bytes:
    raw = value.encode("utf-8", errors="ignore")
    return raw[:24].ljust(24, b"\x00")


class UdpControlClient:
    MORAI_CTRL_HEADER_NAME = b"MoraiCtrlCmd"
    MORAI_CTRL_AUX_HEADER_SIZE = 12
    MORAI_CTRL_TRAILER = b"\r\n"

    def __init__(
        self,
        host: str,
        port: int,
        mode: str = "double3",
        entity_id: str = "EGO",
        bind_host: str = "",
        bind_port: int = 0,
    ) -> None:
        self.address = (host, port)
        self.mode = mode
        self.entity_id = entity_id
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.socket: socket.socket | None = None

    def send(self, command: ControlCommand) -> None:
        if self.socket is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.bind_port:
                bind_host = self.bind_host or "0.0.0.0"
                self.socket.bind((bind_host, self.bind_port))
        payload = self._encode(command.clipped())
        self.socket.sendto(payload, self.address)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _encode(self, command: ControlCommand) -> bytes:
        if self.mode == "double3":
            return struct.pack("<ddd", command.throttle, command.brake, command.steering)
        if self.mode == "double3_with_id":
            return struct.pack(
                "<24sddd",
                _pad_id_24(self.entity_id),
                command.throttle,
                command.brake,
                command.steering,
            )
        if self.mode == "json":
            return json.dumps(
                {
                    "id": self.entity_id,
                    "throttle": command.throttle,
                    "brake": command.brake,
                    "steering_wheel_angle": command.steering,
                }
            ).encode("utf-8")
        if self.mode == "morai_ctrl_cmd":
            payload = struct.pack(
                "<BBBfffff",
                int(command.ctrl_mode),
                int(command.gear),
                int(command.long_cmd_type),
                float(command.velocity_kph),
                float(command.acceleration_mps2),
                float(command.throttle),
                float(command.brake),
                max(-1.0, min(1.0, float(command.steering))),
            )
            return (
                b"#"
                + self.MORAI_CTRL_HEADER_NAME
                + b"$"
                + struct.pack("<I", len(payload))
                + (b"\x00" * self.MORAI_CTRL_AUX_HEADER_SIZE)
                + payload
                + self.MORAI_CTRL_TRAILER
            )
        raise ValueError(f"unsupported control mode: {self.mode}")
