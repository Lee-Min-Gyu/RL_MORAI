from __future__ import annotations

import argparse
import time

from morai_rl.config.runtime import load_config
from morai_rl.core.types import ControlCommand
from morai_rl.io.ros_sync import RosControlClient, RosVehicleStatusReceiver


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check MORAI ROS synchronous-mode control/tick wiring without env reset."
    )
    parser.add_argument("--config", default="morai_rl/stage1_ros_sync_config.toml")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--throttle", type=float, default=0.2)
    parser.add_argument("--brake", type=float, default=0.0)
    parser.add_argument("--steering", type=float, default=0.0)
    parser.add_argument("--zero-first-steps", type=int, default=5)
    parser.add_argument("--zero-last-steps", type=int, default=5)
    parser.add_argument("--ctrl-mode", type=int, default=2)
    parser.add_argument("--gear", type=int, default=4)
    parser.add_argument("--long-cmd-type", type=int, default=1)
    parser.add_argument("--velocity-kph", type=float, default=0.0)
    parser.add_argument("--acceleration-mps2", type=float, default=0.0)
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.ros.enabled:
        raise RuntimeError("this check requires [ros] enabled = true")

    receiver = RosVehicleStatusReceiver(
        topic=config.ros.ego_topic,
        entity_id=config.ros.entity_id,
        imu_topic=config.ros.imu_topic,
        node_name=config.ros.node_name,
        anonymous=config.ros.anonymous,
    )
    control = RosControlClient(
        ctrl_topic=config.ros.ctrl_topic,
        use_sync_mode=config.ros.use_sync_mode,
        user_id=config.ros.user_id,
        time_step=config.ros.time_step,
        sensor_capture=config.ros.sensor_capture,
        sync_info_topic=config.ros.sync_info_topic,
        sync_mode_cmd_service=config.ros.sync_mode_cmd_service,
        sync_ctrl_cmd_service=config.ros.sync_ctrl_cmd_service,
        sync_set_gear_service=config.ros.sync_set_gear_service,
        wait_for_tick_service=config.ros.wait_for_tick_service,
        service_timeout_sec=config.ros.service_timeout_sec,
        wait_for_tick_timeout_sec=config.ros.wait_for_tick_timeout_sec,
        start_sync_on_start=config.ros.start_sync_on_start,
        stop_sync_on_close=config.ros.stop_sync_on_close,
        front_steer_command_scale=config.ros.front_steer_command_scale,
        front_steer_command_sign=config.ros.front_steer_command_sign,
        node_name=config.ros.node_name,
        anonymous=config.ros.anonymous,
    )
    control.attach_vehicle_receiver(receiver)
    receiver.start()

    drive_command = ControlCommand(
        throttle=args.throttle,
        brake=args.brake,
        steering=args.steering,
        ctrl_mode=args.ctrl_mode,
        gear=args.gear,
        long_cmd_type=args.long_cmd_type,
        velocity_kph=args.velocity_kph,
        acceleration_mps2=args.acceleration_mps2,
    )

    try:
        for step in range(args.steps):
            if step < args.zero_first_steps or step >= args.steps - args.zero_last_steps:
                command = ControlCommand.zero()
            else:
                command = drive_command

            control.send(command)
            state = receiver.get_latest()
            if state is None:
                print(f"step={step:03d} frame={control.frame} waiting_for_vehicle_status")
                time.sleep(0.02)
                continue

            print(
                f"step={step:03d} frame={control.frame} "
                f"cmd=({command.throttle:.2f},{command.brake:.2f},{command.steering:+.2f}) "
                f"pos=({state.x:.2f},{state.y:.2f},{state.z:.2f}) "
                f"yaw={state.yaw_deg:.2f} yaw_rate={state.wz:+.4f} "
                f"speed={state.speed_mps:.2f}"
            )
    finally:
        for _ in range(max(1, args.zero_last_steps)):
            try:
                control.send(ControlCommand.zero())
            except Exception:
                break
        receiver.stop()
        control.close()


if __name__ == "__main__":
    main()
