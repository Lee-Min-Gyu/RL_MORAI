from __future__ import annotations

import socket
import struct
import threading
from typing import Optional

from morai_rl.core.types import ObjectState


class ObjectStatusReceiver:
    """Optional receiver for MORAI GT/object sensor packets."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.header_format = "<qii"
        self.object_format = "24si" + "f" * 18
        self.header_size = struct.calcsize(self.header_format)
        self.object_size = struct.calcsize(self.object_format)
        self.socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: list[ObjectState] = []

    def start(self) -> None:
        if self._running:
            return
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(0.2)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_latest(self) -> list[ObjectState]:
        with self._lock:
            return list(self._latest)

    def _loop(self) -> None:
        assert self.socket is not None

        while self._running:
            try:
                payload, _addr = self.socket.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(payload) < self.header_size:
                continue

            seconds, nanos, count = struct.unpack(
                self.header_format, payload[: self.header_size]
            )
            offset = self.header_size
            objects: list[ObjectState] = []
            for _ in range(count):
                chunk = payload[offset : offset + self.object_size]
                if len(chunk) != self.object_size:
                    break
                unpacked = struct.unpack(self.object_format, chunk)
                objects.append(
                    ObjectState(
                        entity_id=unpacked[0].decode("utf-8", errors="ignore").rstrip("\x00"),
                        object_type=int(unpacked[1]),
                        x=float(unpacked[2]),
                        y=float(unpacked[3]),
                        z=float(unpacked[4]),
                        yaw_deg=float(unpacked[7]),
                        length=float(unpacked[8]),
                        width=float(unpacked[9]),
                        height=float(unpacked[10]),
                        vx=float(unpacked[14]),
                        vy=float(unpacked[15]),
                        vz=float(unpacked[16]),
                    )
                )
                offset += self.object_size

            _ = seconds, nanos
            with self._lock:
                self._latest = objects

