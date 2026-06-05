from __future__ import annotations

import pickle
import socket
import struct
from typing import Any


_HEADER = struct.Struct("!Q")


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(payload)))
    sock.sendall(payload)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, _HEADER.size)
    if not header:
        raise EOFError("socket closed while reading message header")
    (size,) = _HEADER.unpack(header)
    payload = _recv_exact(sock, size)
    if len(payload) != size:
        raise EOFError("socket closed while reading message payload")
    message = pickle.loads(payload)
    if not isinstance(message, dict):
        raise TypeError(f"expected dict message, got {type(message).__name__}")
    return message


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

