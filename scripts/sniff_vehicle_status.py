from __future__ import annotations

import argparse
import socket
import struct


def main() -> None:
    parser = argparse.ArgumentParser(description="Sniff raw MORAI vehicle status UDP packets.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9093)
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of packets to print. Use 0 to keep receiving forever.",
    )
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))

    print(f"listening on {args.host}:{args.port}")
    try:
        index = 0
        while args.count <= 0 or index < args.count:
            data, addr = sock.recvfrom(4096)
            preview = data[:32].hex(" ")
            print(f"[{index:02d}] from={addr} len={len(data)} preview={preview}")
            if len(data) == 181 and data[:1] == b"#" and data[10:11] == b"$":
                header_name = data[1:10].decode("ascii", errors="replace")
                payload_len = struct.unpack_from("<I", data, 11)[0]
                print(f"     header={header_name} payload_len={payload_len}")
            if len(data) >= 15 and data[:1] == b"#" and data[1:10] == b"MoraiInfo" and data[10:11] == b"$":
                payload_len = struct.unpack_from("<I", data, 11)[0]
                print(f"     header=MoraiInfo payload_len={payload_len}")
            index += 1
    finally:
        sock.close()


if __name__ == "__main__":
    main()
