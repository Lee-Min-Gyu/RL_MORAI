from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    gym = None
    PPO = None
    Monitor = None
    BaseCallback = None
    CallbackList = None
    CheckpointCallback = None
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None

try:
    import pygame
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    pygame = None
    _PYGAME_IMPORT_ERROR = exc
else:
    _PYGAME_IMPORT_ERROR = None

from morai_rl.envs.gym_wrapper import GymMoraiEnv

CHANNEL_COLORS = {
    "corridor_area": np.array([110, 110, 110], dtype=np.uint8),
    "corridor_boundary": np.array([240, 80, 80], dtype=np.uint8),
    "lane_marking": np.array([250, 220, 90], dtype=np.uint8),
    "reference_centerline": np.array([80, 170, 255], dtype=np.uint8),
    "ego_footprint": np.array([255, 220, 0], dtype=np.uint8),
}


class EpisodeResetStatsCallback(BaseCallback if BaseCallback is not None else object):
    def __init__(self) -> None:
        if BaseCallback is None:  # pragma: no cover - runtime guard
            raise ModuleNotFoundError("stable-baselines3 callbacks are unavailable")
        super().__init__()
        self.episode_count = 0
        self._last_episode_wall_time: float | None = None
        self._episode_term_sums: list[dict[str, float]] = []

    def _ensure_term_buffers(self, env_count: int) -> None:
        while len(self._episode_term_sums) < env_count:
            self._episode_term_sums.append({})

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        self._ensure_term_buffers(len(infos))

        for env_index, (done, info) in enumerate(zip(dones, infos)):
            reward_terms = info.get("reward_terms")
            if isinstance(reward_terms, dict):
                term_sums = self._episode_term_sums[env_index]
                for key, value in reward_terms.items():
                    if isinstance(value, (int, float)):
                        term_sums[key] = term_sums.get(key, 0.0) + float(value)

            if not done:
                continue
            self.episode_count += 1
            now = time.monotonic()
            since_prev_episode = (
                None if self._last_episode_wall_time is None else now - self._last_episode_wall_time
            )
            self._last_episode_wall_time = now
            reason = info.get("termination_reason")
            steps = info.get("step_count")
            progress_m = info.get("episode_progress_m")
            scenario_name = info.get("scenario_name")
            duration_sec = info.get("previous_episode_duration_sec", info.get("episode_duration_sec"))
            episode_reward = None
            episode_info = info.get("episode")
            if isinstance(episode_info, dict):
                episode_reward = episode_info.get("r")
            reward_terms = dict(self._episode_term_sums[env_index])
            self._episode_term_sums[env_index] = {}
            print(
                "episode_end "
                f"count={self.episode_count} "
                f"total_timesteps={self.num_timesteps} "
                f"scenario={scenario_name} "
                f"reason={reason} "
                f"steps={steps} "
                f"progress_m={float(progress_m):+.2f}"
            )
            if episode_reward is not None:
                try:
                    print(f"  total_reward={float(episode_reward):+.3f}")
                except (TypeError, ValueError):
                    print(f"  total_reward={episode_reward}")
            if isinstance(reward_terms, dict) and reward_terms:
                print("  reward_terms")
                for key, value in reward_terms.items():
                    if isinstance(value, (int, float)):
                        signed_value = float(value)
                        if key.endswith("_penalty"):
                            signed_value = -signed_value
                        print(f"    {key}={signed_value:+.3f}")
                    else:
                        print(f"    {key}={value}")
        return True


class TrainingBeVViewerCallback(BaseCallback if BaseCallback is not None else object):
    def __init__(
        self,
        env: GymMoraiEnv,
        scale: int = 6,
        fps: int = 15,
    ) -> None:
        if BaseCallback is None:  # pragma: no cover - runtime guard
            raise ModuleNotFoundError("stable-baselines3 callbacks are unavailable")
        if pygame is None:  # pragma: no cover - runtime guard
            raise ModuleNotFoundError(
                "pygame is required for the training BeV viewer. Install it with `pip install pygame`."
            ) from _PYGAME_IMPORT_ERROR
        super().__init__()
        self.env_ref = env
        self.scale = max(1, int(scale))
        self.fps = max(1, int(fps))
        self._screen = None
        self._font = None
        self._small_font = None
        self._last_draw_time = 0.0
        self._closed = False

    def _on_training_start(self) -> None:
        snapshot = self.env_ref.get_latest_viewer_snapshot()
        bev = snapshot["bev"]
        channel_names = snapshot["channel_names"]
        if bev is None or not channel_names:
            print("BeV viewer disabled: latest observation is not BeV/hybrid.")
            self._closed = True
            return

        pygame.init()
        pygame.font.init()
        height_px, width_px = int(bev.shape[1]), int(bev.shape[2])
        window_width = width_px * self.scale
        window_height = height_px * self.scale + 84
        self._screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("MORAI PPO Training BeV Viewer")
        self._font = pygame.font.SysFont("Consolas", 18)
        self._small_font = pygame.font.SysFont("Consolas", 14)

    def _on_step(self) -> bool:
        if self._closed:
            return True
        assert self._screen is not None
        assert self._font is not None
        assert self._small_font is not None

        now = time.monotonic()
        if now - self._last_draw_time < 1.0 / float(self.fps):
            return True
        self._last_draw_time = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True
                pygame.quit()
                return True

        snapshot = self.env_ref.get_latest_viewer_snapshot()
        bev = snapshot["bev"]
        channel_names = snapshot["channel_names"]
        info = snapshot["info"]
        if bev is None or not channel_names:
            return True

        rgb = np.zeros((bev.shape[1], bev.shape[2], 3), dtype=np.uint8)
        for channel_index, channel_name in enumerate(channel_names):
            mask = bev[channel_index] > 0
            if not np.any(mask):
                continue
            color = CHANNEL_COLORS.get(channel_name, np.array([200, 200, 200], dtype=np.uint8))
            rgb[mask] = np.maximum(rgb[mask], color)

        upscaled = np.kron(rgb, np.ones((self.scale, self.scale, 1), dtype=np.uint8))
        surface = pygame.surfarray.make_surface(np.transpose(upscaled, (1, 0, 2)))

        self._screen.fill((18, 18, 18))
        self._screen.blit(surface, (0, 0))

        state = info.get("state") or {}
        reason = info.get("termination_reason")
        progress_m = info.get("episode_progress_m", 0.0)
        step_count = info.get("step_count", 0)
        line_1 = (
            f"step={step_count} progress={float(progress_m):+.2f}m "
            f"speed={float(state.get('speed_mps', 0.0)):.2f} m/s "
            f"throttle={float(state.get('throttle', 0.0)):.2f} "
            f"brake={float(state.get('brake', 0.0)):.2f}"
        )
        line_2 = (
            f"scenario={info.get('scenario_name')} reason={reason} "
            f"mode={self.env_ref.observation_mode}"
        )
        line_3 = "overlay: corridor/boundary/reference/ego   close window to hide viewer"
        self._screen.blit(self._font.render(line_1, True, (240, 240, 240)), (12, bev.shape[1] * self.scale + 10))
        self._screen.blit(self._small_font.render(line_2, True, (190, 190, 190)), (12, bev.shape[1] * self.scale + 38))
        self._screen.blit(self._small_font.render(line_3, True, (160, 160, 160)), (12, bev.shape[1] * self.scale + 60))
        pygame.display.flip()
        return True

    def _on_training_end(self) -> None:
        if not self._closed and pygame is not None:
            pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO policy on the MORAI RL environment.")
    parser.add_argument("--config", default="morai_rl/example_config.toml")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--save-dir", default="runs/ppo_morai")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy", default="auto")
    parser.add_argument("--sb3-verbose", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--show-bev", action="store_true")
    parser.add_argument("--bev-scale", type=int, default=6)
    parser.add_argument("--bev-fps", type=int, default=15)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--max-restarts", type=int, default=100)
    parser.add_argument("--restart-wait-sec", type=float, default=2.0)
    return parser.parse_args()


def _resolve_resume_path(resume_from: str) -> Path:
    resume_path = Path(resume_from)
    if resume_path.is_file():
        return resume_path
    if resume_path.suffix != ".zip" and resume_path.with_suffix(".zip").is_file():
        return resume_path.with_suffix(".zip")
    raise FileNotFoundError(f"resume model not found: {resume_from}")


def _resolve_policy_name(args: argparse.Namespace, env) -> str:
    if args.policy != "auto":
        return args.policy
    if gym is not None and isinstance(env.observation_space, gym.spaces.Dict):
        return "MultiInputPolicy"
    if len(getattr(env.observation_space, "shape", ())) == 3:
        return "CnnPolicy"
    return "MlpPolicy"


def _build_model(args: argparse.Namespace, env, save_dir: Path):
    policy_name = _resolve_policy_name(args, env)
    print(f"policy={policy_name}")
    if args.resume_from:
        resume_path = _resolve_resume_path(args.resume_from)
        print(f"resuming_from={resume_path}")
        model = PPO.load(
            str(resume_path),
            env=env,
            device=args.device,
        )
        model.verbose = args.sb3_verbose
        model.tensorboard_log = str(save_dir / "tb")
        return model

    return PPO(
        policy=policy_name,
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=args.sb3_verbose,
        tensorboard_log=str(save_dir / "tb"),
        device=args.device,
    )


def _remaining_timesteps(target_timesteps: int, completed_timesteps: int) -> int:
    return max(0, int(target_timesteps) - max(0, int(completed_timesteps)))


def main() -> None:
    if (
        PPO is None
        or Monitor is None
        or CheckpointCallback is None
        or CallbackList is None
    ):  # pragma: no cover - runtime guard
        raise ModuleNotFoundError(
            "stable-baselines3 and torch are required. Install them with "
            "`pip install stable-baselines3 gymnasium torch`."
        ) from _SB3_IMPORT_ERROR

    args = parse_args()
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base_env = GymMoraiEnv(args.config)
    env = Monitor(base_env)
    model = _build_model(args, env, save_dir)
    restart_count = 0
    current_resume_from = args.resume_from
    reset_num_timesteps = not bool(current_resume_from)

    while True:
        remaining_timesteps = _remaining_timesteps(args.timesteps, model.num_timesteps)
        if remaining_timesteps <= 0:
            print(
                "target_timesteps_already_reached "
                f"target={args.timesteps} completed={model.num_timesteps}"
            )
            model.save(str(save_dir / "ppo_model"))
            print(f"saved_model={save_dir / 'ppo_model'}")
            break

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_checkpoint",
        )
        episode_stats_callback = EpisodeResetStatsCallback()
        callbacks = [checkpoint_callback, episode_stats_callback]
        if args.show_bev:
            callbacks.append(
                TrainingBeVViewerCallback(
                    env=base_env,
                    scale=args.bev_scale,
                    fps=args.bev_fps,
                )
            )
        callback = CallbackList(callbacks)
        try:
            print(
                "training_budget "
                f"target={args.timesteps} completed={model.num_timesteps} "
                f"remaining={remaining_timesteps}"
            )
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callback,
                progress_bar=args.progress_bar,
                reset_num_timesteps=reset_num_timesteps,
            )
            model.save(str(save_dir / "ppo_model"))
            print(f"saved_model={save_dir / 'ppo_model'}")
            break
        except KeyboardInterrupt:
            model.save(str(save_dir / "ppo_model_interrupted"))
            print(f"saved_model={save_dir / 'ppo_model_interrupted'}")
            raise
        except Exception as exc:
            model.save(str(save_dir / "ppo_model_crash"))
            print(f"saved_model={save_dir / 'ppo_model_crash'}")
            restart_count += 1
            if restart_count > args.max_restarts:
                raise RuntimeError(
                    f"training aborted after {restart_count} restarts"
                ) from exc

            completed_timesteps = model.num_timesteps
            remaining_timesteps = _remaining_timesteps(args.timesteps, completed_timesteps)
            print(
                "training crashed, restarting from latest crash model "
                f"({restart_count}/{args.max_restarts}, "
                f"num_timesteps={completed_timesteps}, "
                f"remaining_timesteps={remaining_timesteps}, error={exc})"
            )

            env.close()
            time.sleep(args.restart_wait_sec)
            current_resume_from = str(save_dir / "ppo_model_crash.zip")
            args.resume_from = current_resume_from
            base_env = GymMoraiEnv(args.config)
            env = Monitor(base_env)
            model = _build_model(args, env, save_dir)
            reset_num_timesteps = False

    env.close()


if __name__ == "__main__":
    main()
