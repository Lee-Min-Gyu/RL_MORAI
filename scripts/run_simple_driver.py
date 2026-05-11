from __future__ import annotations

import argparse

from morai_rl.baselines.simple_driver import SimpleLaneFollower
from morai_rl.envs.morai_env import MoraiRLEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive the car using the simple rule-based baseline.")
    parser.add_argument("--config", default="morai_rl/example_config.toml")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    env = MoraiRLEnv.from_toml(args.config)
    policy = SimpleLaneFollower()

    try:
        for episode in range(args.episodes):
            _obs, info = env.reset()
            print(f"episode={episode} reset projection={info['projection']}")
            print(f"episode={episode} reset progress=episode_progress={info['episode_progress_m']:.2f}")
            if info.get("corridor") is not None:
                print(f"episode={episode} reset corridor={info['corridor']}")

            while True:
                action = policy.act(info["observation_named"])
                _obs, reward, terminated, truncated, info = env.step(action)
                corridor = info.get("corridor")
                corridor_text = ""
                if corridor is not None:
                    corridor_text = (
                        f" corridor={corridor['corridor_distance_m']:+.3f}"
                        f" inside={corridor['inside']}"
                    )
                print(
                    f"step={env.step_count:04d} reward={reward:+.3f} "
                    f"speed={info['state']['speed_mps']:.2f} "
                    f"dp={info['progress_delta_m']:+.3f} "
                    f"ep={info['episode_progress_m']:+.2f} "
                    f"steer={action.steering:+.3f} "
                    f"lat={info['projection']['lateral_error_m']:+.3f}"
                    f"{corridor_text}"
                )
                if terminated or truncated:
                    print(f"episode={episode} done reason={info['termination_reason']}")
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
