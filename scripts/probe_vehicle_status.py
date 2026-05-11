from __future__ import annotations

import argparse
import math
import socket
import struct


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump raw MoraiInfo payload offsets as float32/float64 candidates."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-bytes", type=int, default=160)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"waiting on {args.host}:{args.port} ...")

    try:
        data, addr = sock.recvfrom(4096)
    finally:
        sock.close()

    print(f"from={addr} len={len(data)}")
    print(f"head={data[:32].hex(' ')}")

    if not (len(data) >= 15 and data[:1] == b"#" and data[1:10] == b"MoraiInfo" and data[10:11] == b"$"):
        print("not a MoraiInfo packet")
        return

    payload_len = struct.unpack_from("<I", data, 11)[0]
    payload = data[27 : 27 + payload_len]
    print(f"payload_len={payload_len}")
    print(f"payload_head={payload[:64].hex(' ')}")

    try:
        seconds, nanos = struct.unpack_from("<ii", payload, 0)
        print(f"timestamp={seconds}.{nanos:09d}")
    except struct.error:
        pass

    limit = min(args.max_bytes, len(payload))

    print("\n[f32 candidates]")
    for offset in range(0, limit - 3, 4):
        value = struct.unpack_from("<f", payload, offset)[0]
        if math.isfinite(value) and abs(value) < 1e6:
            print(f"offset={offset:03d} f32={value}")

    print("\n[f64 candidates]")
    for offset in range(0, limit - 7, 8):
        value = struct.unpack_from("<d", payload, offset)[0]
        if math.isfinite(value) and abs(value) < 1e9:
            print(f"offset={offset:03d} f64={value}")


if __name__ == "__main__":
    main()
