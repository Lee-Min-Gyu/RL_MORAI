from __future__ import annotations

import argparse
import time

from morai_rl.config.runtime import load_config
from morai_rl.io.collision_udp import CollisionStatusReceiver


def _format_entry(index: int, entry) -> str:
    return (
        f"[{index}] type={entry.object_type_name}({entry.object_type}) "
        f"id={entry.object_id} "
        f"pos=({entry.x:.3f}, {entry.y:.3f}, {entry.z:.3f}) "
        f"global_offset=({entry.global_offset_x:.3f}, {entry.global_offset_y:.3f}, {entry.global_offset_z:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MORAI UDP collision packets.")
    parser.add_argument("--config", default="morai_rl/example_config.toml")
    parser.add_argument(
        "--host",
        default=None,
        help="Bind host override. Use 0.0.0.0 if MORAI is on another machine.",
    )
    parser.add_argument("--port", type=int, default=None, help="Collision destination port.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="How long to print collisions. Use 0 to keep receiving forever.",
    )
    parser.add_argument(
        "--show-empty",
        action="store_true",
        help="Also print empty collision slots.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    host = args.host if args.host is not None else config.udp.host
    port = args.port if args.port is not None else config.udp.collision_port

    receiver = CollisionStatusReceiver(host, port)
    receiver.start()

    print(f"listening for collision packets on {host}:{port}")
    try:
        deadline = None if args.duration <= 0 else time.monotonic() + args.duration
        last_timestamp: float | None = None
        while deadline is None or time.monotonic() < deadline:
            state = receiver.get_latest()
            debug = receiver.get_debug_snapshot()
            if state is None:
                if debug["last_packet_len"] is None:
                    print("waiting for collision packet... no packet received yet")
                else:
                    print(
                        "waiting for collision packet... "
                        f"last_packet_len={debug['last_packet_len']} "
                        f"last_payload_len={debug['last_payload_len']} "
                        f"header={debug['last_header_name']} "
                        f"last_error={debug['last_error']}"
                    )
                time.sleep(0.5)
                continue

            if last_timestamp == state.timestamp_sec:
                time.sleep(0.1)
                continue

            last_timestamp = state.timestamp_sec
            print(
                f"collision timestamp={state.seconds}.{state.nanos:09d} "
                f"packet_len={state.packet_len} payload_len={state.payload_len} "
                f"header={state.header_name}"
            )

            visible_count = 0
            for index, entry in enumerate(state.collisions):
                if entry.is_empty and not args.show_empty:
                    continue
                print("  " + _format_entry(index, entry))
                visible_count += 1

            if visible_count == 0:
                print("  no non-empty collision entries in packet")

            time.sleep(0.1)
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
