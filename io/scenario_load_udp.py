from __future__ import annotations

import socket
import struct


class ScenarioLoadClient:
    HEADER_NAME = b"LoadScenario"
    AUX_HEADER_SIZE = 12
    TRAILER = b"\r\n"
    FILE_NAME_SIZE = 30
    PAYLOAD_FORMAT = "<30s???????"

    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        destination_host: str,
        destination_port: int,
        *,
        file_name: str = "",
        delete_all: bool = True,
        load_network_connection_data: bool = True,
        load_ego_vehicle_data: bool = True,
        load_surrounding_vehicle_data: bool = True,
        load_pedestrian_data: bool = True,
        load_object_data: bool = True,
        set_pause: bool = False,
    ) -> None:
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.destination = (destination_host, destination_port)
        self.file_name = file_name
        self.delete_all = delete_all
        self.load_network_connection_data = load_network_connection_data
        self.load_ego_vehicle_data = load_ego_vehicle_data
        self.load_surrounding_vehicle_data = load_surrounding_vehicle_data
        self.load_pedestrian_data = load_pedestrian_data
        self.load_object_data = load_object_data
        self.set_pause = set_pause
        self.socket: socket.socket | None = None

    def send(
        self,
        file_name: str | None = None,
        *,
        delete_all: bool | None = None,
        load_network_connection_data: bool | None = None,
        load_ego_vehicle_data: bool | None = None,
        load_surrounding_vehicle_data: bool | None = None,
        load_pedestrian_data: bool | None = None,
        load_object_data: bool | None = None,
        set_pause: bool | None = None,
    ) -> None:
        if self.socket is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.bind_port:
                self.socket.bind((self.bind_host, self.bind_port))

        resolved_file_name = file_name if file_name is not None else self.file_name
        if not resolved_file_name.strip():
            raise ValueError("scenario file name is empty")

        payload = self._encode_payload(
            file_name=resolved_file_name,
            delete_all=self.delete_all if delete_all is None else delete_all,
            load_network_connection_data=(
                self.load_network_connection_data
                if load_network_connection_data is None
                else load_network_connection_data
            ),
            load_ego_vehicle_data=(
                self.load_ego_vehicle_data
                if load_ego_vehicle_data is None
                else load_ego_vehicle_data
            ),
            load_surrounding_vehicle_data=(
                self.load_surrounding_vehicle_data
                if load_surrounding_vehicle_data is None
                else load_surrounding_vehicle_data
            ),
            load_pedestrian_data=(
                self.load_pedestrian_data
                if load_pedestrian_data is None
                else load_pedestrian_data
            ),
            load_object_data=(
                self.load_object_data
                if load_object_data is None
                else load_object_data
            ),
            set_pause=self.set_pause if set_pause is None else set_pause,
        )
        packet = self._wrap_payload(payload)
        self.socket.sendto(packet, self.destination)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    @classmethod
    def _encode_payload(
        cls,
        *,
        file_name: str,
        delete_all: bool,
        load_network_connection_data: bool,
        load_ego_vehicle_data: bool,
        load_surrounding_vehicle_data: bool,
        load_pedestrian_data: bool,
        load_object_data: bool,
        set_pause: bool,
    ) -> bytes:
        normalized_name = file_name.strip()
        if normalized_name.endswith(".json"):
            normalized_name = normalized_name[:-5]
        name_bytes = normalized_name.encode("utf-8", errors="ignore")
        if len(name_bytes) > cls.FILE_NAME_SIZE:
            raise ValueError(
                f"scenario file name must be <= {cls.FILE_NAME_SIZE} bytes without .json"
            )

        padded_name = name_bytes.ljust(cls.FILE_NAME_SIZE, b" ")
        return struct.pack(
            cls.PAYLOAD_FORMAT,
            padded_name,
            bool(delete_all),
            bool(load_network_connection_data),
            bool(load_ego_vehicle_data),
            bool(load_surrounding_vehicle_data),
            bool(load_pedestrian_data),
            bool(load_object_data),
            bool(set_pause),
        )

    @classmethod
    def _wrap_payload(cls, payload: bytes) -> bytes:
        return (
            b"#"
            + cls.HEADER_NAME
            + b"$"
            + struct.pack("<I", len(payload))
            + (b"\x00" * cls.AUX_HEADER_SIZE)
            + payload
            + cls.TRAILER
        )
