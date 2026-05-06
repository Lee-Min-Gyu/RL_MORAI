from __future__ import annotations

import argparse

from morai_rl.core.types import ControlCommand
from morai_rl.envs.morai_env import MoraiRLEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short zero-action env loop.")
    parser.add_argument("--config", default="morai_rl/example_config.toml")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    env = MoraiRLEnv.from_toml(args.config)
    try:
        obs, info = env.reset()
        print(f"reset obs={obs}")
        print(f"reset projection={info['projection']}")
        print(f"reset progress=episode_progress={info['episode_progress_m']:.2f}")
        if info.get("corridor") is not None:
            print(f"reset corridor={info['corridor']}")
        for step in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(ControlCommand.zero())
            corridor = info.get("corridor")
            corridor_text = ""
            if corridor is not None:
                corridor_text = (
                    f" corridor={corridor['corridor_distance_m']:+.2f}"
                    f" inside={corridor['inside']}"
                )
            print(
                f"step={step:03d} reward={reward:+.3f} "
                f"speed={info['state']['speed_mps']:.2f} "
                f"dp={info['progress_delta_m']:+.3f} "
                f"ep={info['episode_progress_m']:+.2f} "
                f"lat={info['projection']['lateral_error_m']:.2f} "
                f"head={info['projection']['heading_error_rad']:.2f}"
                f"{corridor_text}"
            )
            if terminated or truncated:
                print(f"episode ended: {info['termination_reason']}")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
