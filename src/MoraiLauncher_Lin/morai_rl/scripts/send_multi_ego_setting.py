from __future__ import annotations

import argparse
from dataclasses import replace

from morai_rl.config.runtime import load_config
from morai_rl.core.types import VehicleState
from morai_rl.io.multi_ego_setting_udp import MultiEgoSettingClient
from morai_rl.io.vehicle_status_udp import VehicleStatusReceiver


def build_target_state(
    state: VehicleState,
    *,
    x: float | None,
    y: float | None,
    z: float | None,
    roll_deg: float | None,
    pitch_deg: float | None,
    yaw_deg: float | None,
    dx: float,
    dy: float,
    dz: float,
    dyaw_deg: float,
    speed_kph: float | None,
) -> VehicleState:
    target_vx = state.vx
    target_vy = state.vy
    target_vz = state.vz
    if speed_kph is not None:
        target_vx = float(speed_kph) / 3.6
        target_vy = 0.0
        target_vz = 0.0

    return replace(
        state,
        x=float(x) if x is not None else state.x + float(dx),
        y=float(y) if y is not None else state.y + float(dy),
        z=float(z) if z is not None else state.z + float(dz),
        roll_deg=float(roll_deg) if roll_deg is not None else state.roll_deg,
        pitch_deg=float(pitch_deg) if pitch_deg is not None else state.pitch_deg,
        yaw_deg=float(yaw_deg) if yaw_deg is not None else state.yaw_deg + float(dyaw_deg),
        vx=target_vx,
        vy=target_vy,
        vz=target_vz,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one MultiEgoSetting packet based on the current MORAI ego state."
    )
    parser.add_argument("--config", default="morai_rl/stage1_rl8_config.toml")
    parser.add_argument("--x", type=float, default=None, help="Absolute world X target.")
    parser.add_argument("--y", type=float, default=None, help="Absolute world Y target.")
    parser.add_argument("--z", type=float, default=None, help="Absolute world Z target.")
    parser.add_argument("--roll-deg", type=float, default=None, help="Absolute roll target.")
    parser.add_argument("--pitch-deg", type=float, default=None, help="Absolute pitch target.")
    parser.add_argument("--yaw-deg", type=float, default=None, help="Absolute yaw target.")
    parser.add_argument("--dx", type=float, default=0.0, help="Meters to move along world X.")
    parser.add_argument("--dy", type=float, default=0.0, help="Meters to move along world Y.")
    parser.add_argument("--dz", type=float, default=0.0, help="Meters to move along world Z.")
    parser.add_argument("--dyaw-deg", type=float, default=0.0, help="Degrees to add to yaw.")
    parser.add_argument(
        "--speed-kph",
        type=float,
        default=None,
        help="Override target speed in km/h. Omit to keep the current speed.",
    )
    parser.add_argument("--gear", type=int, default=None, help="Override gear mode for the packet.")
    parser.add_argument(
        "--ctrl-mode",
        type=int,
        default=None,
        help="Override control mode for the packet.",
    )
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    args = parser.parse_args()

    config = load_config(args.config)
    receiver = VehicleStatusReceiver(config.udp.host, config.udp.vehicle_status_port)
    client = MultiEgoSettingClient(
        bind_host=config.reset.multi_ego_setting_bind_host,
        bind_port=config.reset.multi_ego_setting_bind_port,
        destination_host=config.reset.multi_ego_setting_destination_host,
        destination_port=config.reset.multi_ego_setting_destination_port,
        ego_index=config.reset.multi_ego_setting_ego_index,
        camera_index=config.reset.multi_ego_setting_camera_index,
        gear=config.reset.multi_ego_setting_gear,
        ctrl_mode=config.reset.multi_ego_setting_ctrl_mode,
        send_repeats=config.reset.multi_ego_setting_send_repeats,
        send_interval_sec=config.reset.multi_ego_setting_send_interval_sec,
    )

    receiver.start()
    try:
        current_state = receiver.wait_for_state(args.timeout_sec)
        target_state = build_target_state(
            current_state,
            x=args.x,
            y=args.y,
            z=args.z,
            roll_deg=args.roll_deg,
            pitch_deg=args.pitch_deg,
            yaw_deg=args.yaw_deg,
            dx=args.dx,
            dy=args.dy,
            dz=args.dz,
            dyaw_deg=args.dyaw_deg,
            speed_kph=args.speed_kph,
        )
        print(
            f"current pos=({current_state.x:.2f}, {current_state.y:.2f}, {current_state.z:.2f}) "
            f"yaw={current_state.yaw_deg:.2f} speed={current_state.speed_mps * 3.6:.2f} kph"
        )
        print(
            f"target  pos=({target_state.x:.2f}, {target_state.y:.2f}, {target_state.z:.2f}) "
            f"rpy=({target_state.roll_deg:.3f}, {target_state.pitch_deg:.3f}, {target_state.yaw_deg:.3f}) "
            f"speed={target_state.speed_mps * 3.6:.2f} kph "
            f"gear={args.gear if args.gear is not None else config.reset.multi_ego_setting_gear} "
            f"ctrl_mode={args.ctrl_mode if args.ctrl_mode is not None else config.reset.multi_ego_setting_ctrl_mode}"
        )
        print(
            "sending MultiEgoSetting "
            f"to {config.reset.multi_ego_setting_destination_host}:"
            f"{config.reset.multi_ego_setting_destination_port}"
        )
        client.send_state(target_state, gear=args.gear, ctrl_mode=args.ctrl_mode)
        print("packet sent")
    finally:
        client.close()
        receiver.stop()


if __name__ == "__main__":
    main()
