from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Optional

from morai_rl.core.types import VehicleState


class VehicleStatusReceiver:
    """Background UDP receiver for MORAI vehicle status packets."""

    LEGACY_PACKET_FORMAT = "<qi24s" + "f" * 18
    LEGACY_PACKET_SIZE = struct.calcsize(LEGACY_PACKET_FORMAT)

    MORAI_INFO_HEADER_NAME = b"MoraiInfo"
    MORAI_INFO_PREFIX_SIZE = 27
    MORAI_INFO_TRAILER_SIZE = 2
    MORAI_INFO_MIN_PAYLOAD_SIZE = 152

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: Optional[VehicleState] = None
        self._last_packet_len: Optional[int] = None
        self._last_payload_len: Optional[int] = None
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

    def get_latest(self) -> Optional[VehicleState]:
        with self._lock:
            return self._latest

    def get_debug_snapshot(self) -> dict[str, int | str | None]:
        with self._lock:
            return {
                "last_packet_len": self._last_packet_len,
                "last_payload_len": self._last_payload_len,
                "last_error": self._last_error,
            }

    def drain_socket(self) -> None:
        if self.socket is None:
            return

        try:
            self.socket.setblocking(False)
            while True:
                self.socket.recvfrom(4096)
        except BlockingIOError:
            pass
        except OSError:
            pass
        finally:
            try:
                self.socket.setblocking(True)
                self.socket.settimeout(0.2)
            except OSError:
                pass

    def wait_for_state(
        self,
        timeout_sec: float,
        min_timestamp_sec: float | None = None,
    ) -> VehicleState:
        sleeper = threading.Event()
        start = time.monotonic()
        while True:
            latest = self.get_latest()
            if latest is not None and (
                min_timestamp_sec is None or latest.timestamp_sec > min_timestamp_sec
            ):
                return latest
            if timeout_sec is not None and time.monotonic() - start >= timeout_sec:
                raise TimeoutError("vehicle state timeout")
            sleeper.wait(0.02)

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

            state = self._parse_payload(payload)
            if state is None:
                continue

            with self._lock:
                self._latest = state
                self._last_error = None

    def _parse_payload(self, payload: bytes) -> Optional[VehicleState]:
        if len(payload) == self.LEGACY_PACKET_SIZE:
            with self._lock:
                self._last_payload_len = self.LEGACY_PACKET_SIZE
            return self._parse_legacy_packet(payload)

        if self._looks_like_morai_info(payload):
            try:
                return self._parse_morai_info_packet(payload)
            except Exception as exc:
                with self._lock:
                    self._last_error = f"MoraiInfo parse failed: {exc}"
                return None

        with self._lock:
            self._last_error = (
                f"unsupported packet size {len(payload)} "
                f"(legacy={self.LEGACY_PACKET_SIZE}, or MoraiInfo header expected)"
            )
        return None

    def _looks_like_morai_info(self, payload: bytes) -> bool:
        return (
            len(payload) >= self.MORAI_INFO_PREFIX_SIZE
            and payload[0:1] == b"#"
            and payload[1:10] == self.MORAI_INFO_HEADER_NAME
            and payload[10:11] == b"$"
        )

    def _parse_legacy_packet(self, payload: bytes) -> VehicleState:
        unpacked = struct.unpack(self.LEGACY_PACKET_FORMAT, payload)
        return VehicleState(
            timestamp_sec=float(unpacked[0]) + float(unpacked[1]) * 1e-9,
            entity_id=unpacked[2].decode("utf-8", errors="ignore").rstrip("\x00"),
            x=float(unpacked[3]),
            y=float(unpacked[4]),
            z=float(unpacked[5]),
            roll_deg=float(unpacked[6]),
            pitch_deg=float(unpacked[7]),
            yaw_deg=float(unpacked[8]),
            vx=float(unpacked[9]),
            vy=float(unpacked[10]),
            vz=float(unpacked[11]),
            ax=float(unpacked[12]),
            ay=float(unpacked[13]),
            az=float(unpacked[14]),
            wx=float(unpacked[15]),
            wy=float(unpacked[16]),
            wz=float(unpacked[17]),
            throttle=float(unpacked[18]),
            brake=float(unpacked[19]),
            steer_angle=float(unpacked[20]),
        )

    def _parse_morai_info_packet(self, payload: bytes) -> VehicleState:
        payload_len = struct.unpack_from("<I", payload, 11)[0]
        if payload_len < self.MORAI_INFO_MIN_PAYLOAD_SIZE:
            raise ValueError(f"payload_len too small: {payload_len}")

        payload_start = self.MORAI_INFO_PREFIX_SIZE
        payload_end = payload_start + payload_len
        min_total = payload_end
        max_total = payload_end + self.MORAI_INFO_TRAILER_SIZE
        if not (min_total <= len(payload) <= max_total):
            raise ValueError(
                f"unexpected total size {len(payload)} for payload_len={payload_len} "
                f"(expected {min_total} or {max_total})"
            )

        data = payload[payload_start:payload_end]
        with self._lock:
            self._last_payload_len = payload_len

        offset = 0
        seconds, nanos = struct.unpack_from("<ii", data, offset)
        offset += 8

        ctrl_mode = data[offset]
        gear = data[offset + 1]
        offset += 2

        signed_speed_kph = struct.unpack_from("<f", data, offset)[0]
        offset += 4
        map_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        throttle = struct.unpack_from("<f", data, offset)[0]
        offset += 4
        brake = struct.unpack_from("<f", data, offset)[0]
        offset += 4

        size_x, size_y, size_z = struct.unpack_from("<fff", data, offset)
        offset += 12
        overhang, wheelbase, rear_overhang = struct.unpack_from("<fff", data, offset)
        offset += 12
        x, y, z = struct.unpack_from("<fff", data, offset)
        offset += 12
        roll_deg, pitch_deg, yaw_deg = struct.unpack_from("<fff", data, offset)
        offset += 12
        vx_kph, vy_kph, vz_kph = struct.unpack_from("<fff", data, offset)
        offset += 12
        wx, wy, wz = struct.unpack_from("<fff", data, offset)
        offset += 12
        ax, ay, az = struct.unpack_from("<fff", data, offset)
        offset += 12
        steer_angle = struct.unpack_from("<f", data, offset)[0]
        offset += 4
        link_id_raw = struct.unpack_from("<38s", data, offset)[0]
        offset += 38

        link_id = link_id_raw.decode("utf-8", errors="ignore").rstrip("\x00 ")
        entity_id = f"EGO[{link_id}]" if link_id else "EGO"

        _ = (
            ctrl_mode,
            gear,
            signed_speed_kph,
            map_id,
            size_x,
            size_y,
            size_z,
            overhang,
            wheelbase,
            rear_overhang,
            payload_len,
            offset,
        )

        return VehicleState(
            timestamp_sec=float(seconds) + float(nanos) * 1e-9,
            entity_id=entity_id,
            x=float(x),
            y=float(y),
            z=float(z),
            roll_deg=float(roll_deg),
            pitch_deg=float(pitch_deg),
            yaw_deg=float(yaw_deg),
            vx=float(vx_kph) ,
            vy=float(vy_kph) ,
            vz=float(vz_kph) ,
            ax=float(ax),
            ay=float(ay),
            az=float(az),
            wx=float(wx),
            wy=float(wy),
            wz=float(wz),
            throttle=float(throttle),
            brake=float(brake),
            steer_angle=float(steer_angle),
            length_m=float(size_x),
            width_m=float(size_y),
        )
