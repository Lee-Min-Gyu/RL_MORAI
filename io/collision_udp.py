from __future__ import annotations

import socket
import struct
import threading
from typing import Optional

from morai_rl.core.types import CollisionObjectState, CollisionState


class CollisionStatusReceiver:
    """Background UDP receiver for MORAI collision packets."""

    RAW_PAYLOAD_SIZE = 148
    TIMESTAMP_FORMAT = "<ii"
    ENTRY_FORMAT = "<hhffffff"
    ENTRY_COUNT = 5
    TRAILER_SIZE = 2
    ENTRY_SIZE = struct.calcsize(ENTRY_FORMAT)
    TIMESTAMP_SIZE = struct.calcsize(TIMESTAMP_FORMAT)

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: Optional[CollisionState] = None
        self._last_packet_len: Optional[int] = None
        self._last_payload_len: Optional[int] = None
        self._last_header_name: Optional[str] = None
        self._last_error: Optional[str] = None

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

    def clear_latest(self) -> None:
        with self._lock:
            self._latest = None

    def get_latest(self) -> Optional[CollisionState]:
        with self._lock:
            return self._latest

    def get_debug_snapshot(self) -> dict[str, int | str | None]:
        with self._lock:
            return {
                "last_packet_len": self._last_packet_len,
                "last_payload_len": self._last_payload_len,
                "last_header_name": self._last_header_name,
                "last_error": self._last_error,
            }

    def _loop(self) -> None:
        assert self.socket is not None

        while self._running:
            try:
                payload, _addr = self.socket.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            with self._lock:
                self._last_packet_len = len(payload)

            state = self._parse_packet(payload)
            if state is None:
                continue

            with self._lock:
                self._latest = state
                self._last_error = None

    def _parse_packet(self, packet: bytes) -> Optional[CollisionState]:
        try:
            payload, header_name, payload_len = self._extract_payload(packet)
            state = self._parse_payload(payload)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            return None

        state.header_name = header_name
        state.packet_len = len(packet)
        state.payload_len = payload_len

        with self._lock:
            self._last_payload_len = payload_len
            self._last_header_name = header_name

        return state

    def _extract_payload(self, packet: bytes) -> tuple[bytes, str | None, int]:
        if len(packet) == self.RAW_PAYLOAD_SIZE:
            return packet, None, self.RAW_PAYLOAD_SIZE

        if len(packet) < 7 or packet[:1] != b"#":
            raise ValueError(
                f"unsupported collision packet size {len(packet)} without MORAI header"
            )

        dollar_index = packet.find(b"$", 1, min(len(packet), 32))
        if dollar_index < 0:
            raise ValueError("collision header terminator '$' not found")

        header_name = packet[1:dollar_index].decode("ascii", errors="ignore").rstrip("\x00 ")
        payload_len_offset = dollar_index + 1
        if len(packet) < payload_len_offset + 4:
            raise ValueError("collision header missing payload length")

        payload_len = struct.unpack_from("<I", packet, payload_len_offset)[0]
        if payload_len != self.RAW_PAYLOAD_SIZE:
            raise ValueError(
                f"unexpected collision payload_len={payload_len} (expected {self.RAW_PAYLOAD_SIZE})"
            )

        for trailer_size in (self.TRAILER_SIZE, 0):
            start = len(packet) - payload_len - trailer_size
            end = start + payload_len
            if start >= payload_len_offset + 4 and end <= len(packet):
                return packet[start:end], header_name, payload_len

        raise ValueError(
            f"could not locate collision payload in packet_len={len(packet)} payload_len={payload_len}"
        )

    def _parse_payload(self, payload: bytes) -> CollisionState:
        if len(payload) != self.RAW_PAYLOAD_SIZE:
            raise ValueError(
                f"unexpected collision payload size {len(payload)} (expected {self.RAW_PAYLOAD_SIZE})"
            )

        seconds, nanos = struct.unpack_from(self.TIMESTAMP_FORMAT, payload, 0)
        offset = self.TIMESTAMP_SIZE
        collisions: list[CollisionObjectState] = []

        for _ in range(self.ENTRY_COUNT):
            unpacked = struct.unpack_from(self.ENTRY_FORMAT, payload, offset)
            collisions.append(
                CollisionObjectState(
                    object_type=int(unpacked[0]),
                    object_id=int(unpacked[1]),
                    x=float(unpacked[2]),
                    y=float(unpacked[3]),
                    z=float(unpacked[4]),
                    global_offset_x=float(unpacked[5]),
                    global_offset_y=float(unpacked[6]),
                    global_offset_z=float(unpacked[7]),
                )
            )
            offset += self.ENTRY_SIZE

        return CollisionState(
            timestamp_sec=float(seconds) + float(nanos) * 1e-9,
            seconds=int(seconds),
            nanos=int(nanos),
            collisions=collisions,
        )
