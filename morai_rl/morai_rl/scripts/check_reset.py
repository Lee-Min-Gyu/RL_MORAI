from __future__ import annotations

import argparse
import math
import time

from morai_rl.envs.morai_env import MoraiRLEnv


def _wrap_angle_deg(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated MORAI ROS2 reset checks.")
    parser.add_argument("--config", default="morai_rl/config/stage1_rl8_ros2.toml")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    args = parser.parse_args()

    env = MoraiRLEnv.from_toml(args.config)
    reset_cfg = env.config.reset
    target_x = reset_cfg.multi_ego_setting_target_x
    target_y = reset_cfg.multi_ego_setting_target_y
    target_z = reset_cfg.multi_ego_setting_target_z
    target_yaw = reset_cfg.multi_ego_setting_target_yaw_deg

    try:
        for index in range(max(1, args.repeats)):
            start = time.monotonic()
            _obs, info = env.reset()
            elapsed = time.monotonic() - start
            state = info["state"]

            position_error = None
            yaw_error = None
            if target_x is not None and target_y is not None and target_z is not None:
                dx = float(state["x"]) - float(target_x)
                dy = float(state["y"]) - float(target_y)
                dz = float(state["z"]) - float(target_z)
                position_error = math.sqrt(dx * dx + dy * dy + dz * dz)
                yaw_error = abs(_wrap_angle_deg(float(state["yaw_deg"]) - float(target_yaw)))

            error_text = ""
            if position_error is not None and yaw_error is not None:
                error_text = f" pos_err={position_error:.3f}m yaw_err={yaw_error:.2f}deg"

            print(
                f"reset={index:03d}"
                f" strategy={info['reset_strategy']}"
                f" stable_frames={info['stable_frames']}"
                f" elapsed={elapsed:.2f}s"
                f" x={state['x']:.3f} y={state['y']:.3f} z={state['z']:.3f}"
                f" yaw={state['yaw_deg']:.3f}"
                f" speed={state['speed_mps']:.3f}"
                f"{error_text}"
            )

            if args.sleep_sec > 0.0 and index + 1 < args.repeats:
                time.sleep(args.sleep_sec)
    finally:
        env.close()


if __name__ == "__main__":
    main()
