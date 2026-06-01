from __future__ import annotations

import argparse
import time

from morai_rl.core.types import ControlCommand
from morai_rl.envs.morai_env import MoraiRLEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated MoraiRLEnv reset checks.")
    parser.add_argument("--config", default="morai_rl/stage1_ros_sync_config.toml")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--drive-steps-before-reset", type=int, default=0, help="Apply a simple drive command before resets after the first one.")
    parser.add_argument("--throttle", type=float, default=1.0)
    parser.add_argument("--brake", type=float, default=0.0)
    parser.add_argument("--steering", type=float, default=0.0)
    parser.add_argument("--gear", type=int, default=4)
    args = parser.parse_args()

    env = MoraiRLEnv.from_toml(args.config)
    repeats = max(1, int(args.repeats))
    drive_steps = max(0, int(args.drive_steps_before_reset))

    try:
        for index in range(repeats):
            if index > 0 and drive_steps > 0:
                command = ControlCommand(
                    throttle=args.throttle,
                    brake=args.brake,
                    steering=args.steering,
                    gear=args.gear,
                )
                for step in range(drive_steps):
                    _, _, terminated, truncated, info = env.step(command)
                    state = info["state"]
                    print(
                        f"pre-reset-drive repeat={index + 1:02d} step={step + 1:03d} "
                        f"pos=({state['x']:.2f},{state['y']:.2f}) "
                        f"speed={state['speed_mps']:.2f}",
                        flush=True,
                    )
                    if terminated or truncated:
                        print(
                            f"pre-reset-drive ended: {info['termination_reason']}",
                            flush=True,
                        )
                        break

            print(f"reset start {index + 1}/{repeats}", flush=True)
            _, info = env.reset()
            state = info["state"]
            projection = info["projection"]
            corridor = info.get("corridor")
            corridor_text = ""
            if corridor is not None:
                corridor_text = (
                    f" corridor={corridor['corridor_distance_m']:+.2f}"
                    f" inside={corridor['inside']}"
                )
            print(
                f"reset ok {index + 1}/{repeats} "
                f"scenario={info.get('scenario_name')} "
                f"strategy={info.get('reset_strategy')} "
                f"stable_frames={info.get('stable_frames')} "
                f"pos=({state['x']:.2f},{state['y']:.2f},{state['z']:.2f}) "
                f"yaw={state['yaw_deg']:.2f} speed={state['speed_mps']:.2f} "
                f"progress={projection['progress_m']:.2f} "
                f"lat={projection['lateral_error_m']:.2f}"
                f"{corridor_text}",
                flush=True,
            )
            if index + 1 < repeats and args.sleep_sec > 0.0:
                time.sleep(args.sleep_sec)
    finally:
        env.close()


if __name__ == "__main__":
    main()
