from __future__ import annotations

import socket
import struct
import time

from morai_rl.core.types import VehicleState


class MultiEgoSettingClient:
    HEADER_NAME = b"MultiEgoSetting"
    AUX_HEADER_SIZE = 12
    TRAILER = b"\r\n"
    MAX_EGO_COUNT = 20
    BODY_FORMAT = "<hfffffffBB"
    BODY_SIZE = struct.calcsize(BODY_FORMAT)

    def __init__(
        self,
        destination_host: str,
        destination_port: int,
        *,
        bind_host: str = "",
        bind_port: int = 0,
        ego_index: int = 0,
        camera_index: int = 0,
        gear: int = 4,
        ctrl_mode: int = 2,
        send_repeats: int = 3,
        send_interval_sec: float = 0.05,
    ) -> None:
        self.destination = (destination_host, destination_port)
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.ego_index = int(ego_index)
        self.camera_index = int(camera_index)
        self.gear = int(gear)
        self.ctrl_mode = int(ctrl_mode)
        self.send_repeats = max(1, int(send_repeats))
        self.send_interval_sec = max(0.0, float(send_interval_sec))
        self.socket: socket.socket | None = None

    def send_state(
        self,
        state: VehicleState,
        *,
        gear: int | None = None,
        ctrl_mode: int | None = None,
    ) -> None:
        payload = self._encode_payload(state, gear=gear, ctrl_mode=ctrl_mode)
        packet = self._wrap_payload(payload)
        sock = self._ensure_socket()
        for repeat_index in range(self.send_repeats):
            sock.sendto(packet, self.destination)
            if repeat_index + 1 < self.send_repeats and self.send_interval_sec > 0.0:
                time.sleep(self.send_interval_sec)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _ensure_socket(self) -> socket.socket:
        if self.socket is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.bind_port:
                bind_host = self.bind_host or "0.0.0.0"
                self.socket.bind((bind_host, self.bind_port))
        return self.socket

    def _encode_payload(
        self,
        state: VehicleState,
        *,
        gear: int | None = None,
        ctrl_mode: int | None = None,
    ) -> bytes:
        speed_kph = float(state.speed_mps) * 3.6
        target_gear = self.gear if gear is None else int(gear)
        target_ctrl_mode = self.ctrl_mode if ctrl_mode is None else int(ctrl_mode)
        bodies = [
            struct.pack(
                self.BODY_FORMAT,
                int(self.ego_index),
                float(state.x),
                float(state.y),
                float(state.z),
                float(state.roll_deg),
                float(state.pitch_deg),
                float(state.yaw_deg),
                float(speed_kph),
                target_gear,
                target_ctrl_mode,
            )
        ]
        zero_body = b"\x00" * self.BODY_SIZE
        while len(bodies) < self.MAX_EGO_COUNT:
            bodies.append(zero_body)
        return struct.pack("<ii", 1, int(self.camera_index)) + b"".join(bodies)

    def _wrap_payload(self, payload: bytes) -> bytes:
        return (
            b"#"
            + self.HEADER_NAME
            + b"$"
            + struct.pack("<I", len(payload))
            + (b"\x00" * self.AUX_HEADER_SIZE)
            + payload
            + self.TRAILER
        )
