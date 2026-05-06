from __future__ import annotations

import argparse
import socket
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure MORAI vehicle-status UDP receive rate.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(args.seconds)

    print(f"measuring on {args.host}:{args.port} for {args.seconds:.1f}s ...")

    count = 0
    start = time.perf_counter()
    end = start + args.seconds
    try:
        while time.perf_counter() < end:
            timeout_left = max(0.01, end - time.perf_counter())
            sock.settimeout(timeout_left)
            try:
                sock.recvfrom(4096)
            except socket.timeout:
                break
            count += 1
    finally:
        sock.close()

    elapsed = time.perf_counter() - start
    hz = count / elapsed if elapsed > 0 else 0.0
    print(f"packets={count} elapsed={elapsed:.3f}s rate={hz:.2f}Hz")


if __name__ == "__main__":
    main()
