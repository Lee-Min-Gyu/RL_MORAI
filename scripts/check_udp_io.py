from __future__ import annotations

import argparse
import time

from morai_rl.config.runtime import load_config
from morai_rl.core.types import ControlCommand
from morai_rl.io.control_udp import UdpControlClient
from morai_rl.io.vehicle_status_udp import VehicleStatusReceiver


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MORAI UDP control/status wiring.")
    parser.add_argument("--config", default="morai_rl/example_config.toml")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="How long to print status. Use 0 to keep receiving forever.",
    )
    parser.add_argument("--send-zero", action="store_true")
    parser.add_argument("--send-command", action="store_true")
    parser.add_argument("--throttle", type=float, default=0.2)
    parser.add_argument("--brake", type=float, default=0.0)
    parser.add_argument("--steering", type=float, default=0.0)
    parser.add_argument("--ctrl-mode", type=int, default=2)
    parser.add_argument("--gear", type=int, default=4)
    parser.add_argument("--long-cmd-type", type=int, default=1)
    parser.add_argument("--velocity-kph", type=float, default=0.0)
    parser.add_argument("--acceleration-mps2", type=float, default=0.0)
    args = parser.parse_args()

    config = load_config(args.config)
    receiver = VehicleStatusReceiver(config.udp.host, config.udp.vehicle_status_port)
    control = UdpControlClient(
        config.udp.host,
        config.udp.control_port,
        config.udp.control_mode,
        config.udp.entity_id,
        bind_host=config.udp.control_bind_host,
        bind_port=config.udp.control_bind_port,
    )
    receiver.start()

    try:
        deadline = None if args.duration <= 0 else time.monotonic() + args.duration
        while deadline is None or time.monotonic() < deadline:
            if args.send_zero:
                control.send(ControlCommand.zero())
            elif args.send_command:
                control.send(
                    ControlCommand(
                        throttle=args.throttle,
                        brake=args.brake,
                        steering=args.steering,
                        ctrl_mode=args.ctrl_mode,
                        gear=args.gear,
                        long_cmd_type=args.long_cmd_type,
                        velocity_kph=args.velocity_kph,
                        acceleration_mps2=args.acceleration_mps2,
                    )
                )
            state = receiver.get_latest()
            debug = receiver.get_debug_snapshot()
            if state is None:
                if debug["last_packet_len"] is None:
                    print("waiting for vehicle status... no packet received yet")
                else:
                    print(
                        "waiting for vehicle status... "
                        f"last_packet_len={debug['last_packet_len']} "
                        f"last_payload_len={debug['last_payload_len']} "
                        f"last_error={debug['last_error']}"
                    )
            else:
                print(
                    f"id={state.entity_id} pos=({state.x:.2f}, {state.y:.2f}) "
                    f"yaw={state.yaw_deg:.2f} speed={state.speed_mps:.2f} "
                    f"last_packet_len={debug['last_packet_len']} "
                    f"last_payload_len={debug['last_payload_len']}"
                )
            time.sleep(0.5)
    finally:
        receiver.stop()
        control.close()


if __name__ == "__main__":
    main()
