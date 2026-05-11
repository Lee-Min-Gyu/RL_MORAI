from __future__ import annotations

import argparse
import time

from morai_rl.core.types import ControlCommand
from morai_rl.envs.morai_env import MoraiRLEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check repeated reset followed by fixed throttle/steering steps."
    )
    parser.add_argument("--config", default="morai_rl/config/stage1_rl8_ros2.toml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--throttle", type=float, default=0.25)
    parser.add_argument("--brake", type=float, default=0.0)
    parser.add_argument("--steering", type=float, default=0.0)
    parser.add_argument("--min-speed-mps", type=float, default=0.3)
    parser.add_argument("--min-progress-m", type=float, default=0.3)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    args = parser.parse_args()

    action = ControlCommand(
        throttle=args.throttle,
        brake=args.brake,
        steering=args.steering,
    )

    env = MoraiRLEnv.from_toml(args.config)
    try:
        for episode in range(max(1, args.episodes)):
            reset_start = time.monotonic()
            _obs, info = env.reset()
            reset_elapsed = time.monotonic() - reset_start
            reset_state = info["state"]
            print(
                f"episode={episode:03d} reset"
                f" elapsed={reset_elapsed:.2f}s"
                f" stable_frames={info['stable_frames']}"
                f" x={reset_state['x']:.3f} y={reset_state['y']:.3f}"
                f" yaw={reset_state['yaw_deg']:.3f}"
                f" speed={reset_state['speed_mps']:.3f}"
            )

            first_speed_step = None
            first_progress_step = None
            final_info = info
            ended = False

            for step in range(max(1, args.steps)):
                _obs, reward, terminated, truncated, final_info = env.step(action)
                speed = float(final_info["state"]["speed_mps"])
                episode_progress = float(final_info["episode_progress_m"])
                if first_speed_step is None and speed >= args.min_speed_mps:
                    first_speed_step = step
                if first_progress_step is None and episode_progress >= args.min_progress_m:
                    first_progress_step = step

                if step == 0 or step == args.steps - 1 or terminated or truncated:
                    print(
                        f"episode={episode:03d} step={step:03d}"
                        f" reward={reward:+.3f}"
                        f" speed={speed:.3f}"
                        f" dp={final_info['progress_delta_m']:+.3f}"
                        f" ep={episode_progress:+.3f}"
                        f" reason={final_info['termination_reason']}"
                    )

                if terminated or truncated:
                    ended = True
                    break

            final_speed = float(final_info["state"]["speed_mps"])
            final_progress = float(final_info["episode_progress_m"])
            ok = (
                first_speed_step is not None
                and first_progress_step is not None
                and not ended
            )
            print(
                f"episode={episode:03d} summary"
                f" ok={ok}"
                f" first_speed_step={first_speed_step}"
                f" first_progress_step={first_progress_step}"
                f" final_speed={final_speed:.3f}"
                f" final_progress={final_progress:.3f}"
            )

            if args.sleep_sec > 0.0 and episode + 1 < args.episodes:
                time.sleep(args.sleep_sec)
    finally:
        env.close()


if __name__ == "__main__":
    main()
