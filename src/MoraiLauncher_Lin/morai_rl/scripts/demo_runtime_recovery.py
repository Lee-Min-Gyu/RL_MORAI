from __future__ import annotations

import argparse
from pathlib import Path

from morai_rl.core.simulator_process import launch_process, terminate_processes_by_name
from morai_rl.core.types import ControlCommand
from morai_rl.envs.morai_env import MoraiRLEnv

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "stage1_ros_sync_config.toml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an RL-style MORAI step/reset loop and intentionally kill the simulator "
            "once so reset recovery can be observed."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--kill-at-step", type=int, default=20)
    parser.add_argument(
        "--process-name",
        action="append",
        default=[],
        help="Simulator process name to kill. Defaults to Simulator.x86_64.",
    )
    parser.add_argument("--relaunch-command", default="")
    parser.add_argument("--relaunch-wait-sec", type=float, default=20.0)
    parser.add_argument("--throttle", type=float, default=0.3)
    parser.add_argument("--brake", type=float, default=0.0)
    parser.add_argument("--steering", type=float, default=0.0)
    parser.add_argument("--gear", type=int, default=4)
    args = parser.parse_args()

    process_names = args.process_name or ["Simulator.x86_64"]
    command = ControlCommand(
        throttle=args.throttle,
        brake=args.brake,
        steering=args.steering,
        gear=args.gear,
    )

    print(f"loading env config: {args.config}", flush=True)
    print(
        "recovery demo "
        f"steps={args.steps} kill_at_step={args.kill_at_step} "
        f"process_names={process_names}",
        flush=True,
    )

    env = MoraiRLEnv.from_toml(args.config)
    total_steps_completed = 0
    episode_index = 0
    killed_once = False

    try:
        print("initial reset start", flush=True)
        _, info = env.reset()
        episode_index += 1
        print_reset_ok("initial reset ok", episode_index, total_steps_completed, info)

        while total_steps_completed < args.steps:
            if not killed_once and total_steps_completed == args.kill_at_step:
                killed_once = True
                killed_pids = terminate_processes_by_name(process_names)
                launched_pid = launch_process(args.relaunch_command) if args.relaunch_command.strip() else None
                print(
                    "injected_simulator_failure "
                    f"at_total_step={total_steps_completed} "
                    f"process_names={process_names} killed_pids={killed_pids} "
                    f"launched_pid={launched_pid}",
                    flush=True,
                )
                if launched_pid is not None and args.relaunch_wait_sec > 0.0:
                    print(f"waiting after relaunch {args.relaunch_wait_sec:.1f}s", flush=True)
                    time.sleep(args.relaunch_wait_sec)

            try:
                _, reward, terminated, truncated, info = env.step(command)
            except TimeoutError as exc:
                print(
                    "step_timeout_seen "
                    f"total_steps_completed={total_steps_completed} error={exc}",
                    flush=True,
                )
                episode_index = reset_or_raise(
                    env=env,
                    label="reset after timeout",
                    episode_index=episode_index,
                    total_steps_completed=total_steps_completed,
                )
                continue
            except RuntimeError as exc:
                print(
                    "runtime_error_seen "
                    f"total_steps_completed={total_steps_completed} error={exc}",
                    flush=True,
                )
                episode_index = reset_or_raise(
                    env=env,
                    label="reset after runtime error",
                    episode_index=episode_index,
                    total_steps_completed=total_steps_completed,
                )
                continue

            total_steps_completed += 1
            state = info["state"]
            print(
                "train_step "
                f"total={total_steps_completed:05d} "
                f"episode={episode_index:03d} "
                f"episode_step={info.get('step_count')} "
                f"reward={reward:+.3f} "
                f"speed={state['speed_mps']:.2f} "
                f"progress={info.get('episode_progress_m', 0.0):+.2f} "
                f"reason={info.get('termination_reason')}",
                flush=True,
            )

            if terminated or truncated:
                print(
                    "episode_done "
                    f"total_steps_completed={total_steps_completed} "
                    f"reason={info.get('termination_reason')}",
                    flush=True,
                )
                episode_index = reset_or_raise(
                    env=env,
                    label="reset after episode done",
                    episode_index=episode_index,
                    total_steps_completed=total_steps_completed,
                )
    finally:
        env.close()


def reset_or_raise(env: MoraiRLEnv, label: str, episode_index: int, total_steps_completed: int) -> int:
    print(f"{label} start", flush=True)
    try:
        _, info = env.reset()
    except Exception as exc:
        print(f"{label} failed error={exc}", flush=True)
        raise
    episode_index += 1
    print_reset_ok(f"{label} ok", episode_index, total_steps_completed, info)
    return episode_index


def print_reset_ok(prefix: str, episode_index: int, total_steps_completed: int, info: dict) -> None:
    state = info["state"]
    print(
        f"{prefix} "
        f"episode={episode_index:03d} "
        f"total_steps_completed={total_steps_completed} "
        f"reset_attempts={info.get('reset_attempts')} "
        f"scenario={info.get('scenario_name')} "
        f"strategy={info.get('reset_strategy')} "
        f"pos=({state['x']:.2f},{state['y']:.2f}) "
        f"speed={state['speed_mps']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
