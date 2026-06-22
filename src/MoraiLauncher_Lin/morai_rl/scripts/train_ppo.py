from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

try:
    import gymnasium as gym
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    gym = None
    th = None
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

from morai_rl.config.runtime import load_config
from morai_rl.core.simulator_process import (
    launch_process,
    terminate_processes_by_cmdline_substrings,
    terminate_processes_by_name,
)
from morai_rl.core.episode_stats import (
    EpisodeStatsWriter,
    ScenarioStatsAccumulator,
    build_episode_summary,
)
from morai_rl.envs.gym_wrapper import GymMoraiEnv
from morai_rl.policies.squashed_policy import (
    SquashedActorCriticCnnPolicy,
    SquashedActorCriticPolicy,
    SquashedMultiInputActorCriticPolicy,
)
from morai_rl.policies.bev_extractor import BeVLightCombinedExtractor

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "stage1_ros_sync_config.toml"

CHANNEL_COLORS = {
    "corridor_area": np.array([110, 110, 110], dtype=np.uint8),
    "corridor_boundary": np.array([240, 80, 80], dtype=np.uint8),
    "lane_marking": np.array([250, 220, 90], dtype=np.uint8),
    "reference_centerline": np.array([80, 170, 255], dtype=np.uint8),
    "ego_footprint": np.array([255, 220, 0], dtype=np.uint8),
}


class EpisodeResetStatsCallback(BaseCallback if BaseCallback is not None else object):
    def __init__(
        self,
        initial_step_log_count: int = 0,
        stats_dir: str | Path | None = None,
        scenario_stats_every: int = 10,
    ) -> None:
        if BaseCallback is None:  # pragma: no cover - runtime guard
            raise ModuleNotFoundError("stable-baselines3 callbacks are unavailable")
        super().__init__()
        self.episode_count = 0
        self.initial_step_log_count = max(0, int(initial_step_log_count))
        self.scenario_stats_every = max(0, int(scenario_stats_every))
        self._writer = EpisodeStatsWriter(stats_dir, prefix="train_episodes") if stats_dir else None
        self._scenario_stats = ScenarioStatsAccumulator()
        self._last_episode_wall_time: float | None = None
        self._episode_term_sums: list[dict[str, float]] = []
        self._episode_reward_sums: list[float] = []

    def _ensure_term_buffers(self, env_count: int) -> None:
        while len(self._episode_term_sums) < env_count:
            self._episode_term_sums.append({})
        while len(self._episode_reward_sums) < env_count:
            self._episode_reward_sums.append(0.0)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        self._ensure_term_buffers(len(infos))
        rewards = self.locals.get("rewards")

        for env_index, (done, info) in enumerate(zip(dones, infos)):
            self._print_initial_step(info)
            if rewards is not None:
                try:
                    self._episode_reward_sums[env_index] += float(rewards[env_index])
                except (IndexError, TypeError, ValueError):
                    pass
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
            self._last_episode_wall_time = now
            reason = info.get("termination_reason")
            steps = info.get("step_count")
            progress_m = info.get("episode_progress_m")
            scenario_name = info.get("scenario_name")
            episode_reward = None
            episode_info = info.get("episode")
            if isinstance(episode_info, dict):
                episode_reward = episode_info.get("r")
            if episode_reward is None:
                episode_reward = self._episode_reward_sums[env_index]
            reward_terms = dict(self._episode_term_sums[env_index])
            self._episode_term_sums[env_index] = {}
            self._episode_reward_sums[env_index] = 0.0
            print(
                "episode_end "
                f"count={self.episode_count} "
                f"total_timesteps={self.num_timesteps} "
                f"scenario={scenario_name} "
                f"reason={reason} "
                f"steps={steps} "
                f"progress_m={float(progress_m):+.2f}",
                flush=True,
            )
            if episode_reward is not None:
                try:
                    print(f"  total_reward={float(episode_reward):+.3f}", flush=True)
                except (TypeError, ValueError):
                    print(f"  total_reward={episode_reward}", flush=True)
            if isinstance(reward_terms, dict) and reward_terms:
                print("  reward_terms", flush=True)
                for key, value in reward_terms.items():
                    if isinstance(value, (int, float)):
                        signed_value = float(value)
                        if key.endswith("_penalty"):
                            signed_value = -signed_value
                        print(f"    {key}={signed_value:+.3f}", flush=True)
                    else:
                        print(f"    {key}={value}", flush=True)
            summary = build_episode_summary(
                source="train",
                episode_index=self.episode_count,
                info=info,
                episode_reward=float(episode_reward),
                reward_terms=reward_terms,
                total_timesteps=int(self.num_timesteps),
            )
            self._scenario_stats.add(summary)
            if self._writer is not None:
                self._writer.write(summary)
            if self.scenario_stats_every > 0 and self.episode_count % self.scenario_stats_every == 0:
                self._scenario_stats.print_summary(
                    label="scenario_stats",
                    total_timesteps=int(self.num_timesteps),
                )
        return True

    def _on_training_end(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def _print_initial_step(self, info: dict) -> None:
        if self.initial_step_log_count <= 0:
            return
        step_count = info.get("step_count")
        if not isinstance(step_count, int) or step_count < 1 or step_count > self.initial_step_log_count:
            return
        projection = info.get("projection") or {}
        corridor = info.get("corridor") or {}
        state = info.get("state") or {}
        progress_m = float(info.get("episode_progress_m", 0.0))
        progress_delta_m = float(info.get("progress_delta_m", 0.0))
        lat = float(projection.get("lateral_error_m", 0.0))
        head = float(projection.get("heading_error_rad", 0.0))
        speed = float(state.get("speed_mps", 0.0))
        yaw = float(state.get("yaw_deg", 0.0))
        print(
            "episode_initial_step "
            f"total_timesteps={self.num_timesteps} "
            f"scenario={info.get('scenario_name')} "
            f"step={step_count:03d} "
            f"progress={progress_m:+.2f} "
            f"dp={progress_delta_m:+.3f} "
            f"lat={lat:+.3f} "
            f"head={head:+.3f} "
            f"speed={speed:.2f} "
            f"yaw={yaw:.2f}",
            flush=True,
        )


class PenaltyCurriculumCallback(BaseCallback if BaseCallback is not None else object):
    STAGES = [
        {"threshold_m": None, "off_track_penalty": 800.0, "stalled_penalty": 800.0},
        {"threshold_m": 500.0, "off_track_penalty": 1200.0, "stalled_penalty": 1200.0},
        {"threshold_m": 900.0, "off_track_penalty": 1600.0, "stalled_penalty": 1600.0},
        {"threshold_m": 1300.0, "off_track_penalty": 2000.0, "stalled_penalty": 2000.0},
    ]

    def __init__(
        self,
        env: GymMoraiEnv,
        state_path: str | Path,
        window_episodes: int = 39,
        required_episodes: int = 34,
    ) -> None:
        if BaseCallback is None:  # pragma: no cover - runtime guard
            raise ModuleNotFoundError("stable-baselines3 callbacks are unavailable")
        super().__init__()
        self.env_ref = env
        self.state_path = Path(state_path)
        self.window_episodes = max(1, int(window_episodes))
        self.required_episodes = max(1, min(int(required_episodes), self.window_episodes))
        self.stage_index = 0
        self.recent_progress_m: deque[float] = deque(maxlen=self.window_episodes)

    def _on_training_start(self) -> None:
        self._load_state()
        self._apply_stage(reason="training_start")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for done, info in zip(dones, infos):
            if not done:
                continue
            try:
                progress_m = float(info.get("episode_progress_m", 0.0))
            except (TypeError, ValueError):
                progress_m = 0.0
            self.recent_progress_m.append(progress_m)
            if self._maybe_advance_stage():
                self._apply_stage(reason="threshold_met")
            self._save_state()
        return True

    def _maybe_advance_stage(self) -> bool:
        next_stage_index = self.stage_index + 1
        if next_stage_index >= len(self.STAGES):
            return False
        if len(self.recent_progress_m) < self.window_episodes:
            return False
        threshold_m = self.STAGES[next_stage_index]["threshold_m"]
        if threshold_m is None:
            return False
        passed = sum(1 for progress_m in self.recent_progress_m if progress_m >= float(threshold_m))
        if passed < self.required_episodes:
            return False
        self.stage_index = next_stage_index
        return True

    def _apply_stage(self, reason: str) -> None:
        stage = self.STAGES[self.stage_index]
        env_config = self.env_ref.env.config.env
        env_config.off_track_penalty = float(stage["off_track_penalty"])
        env_config.stalled_penalty = float(stage["stalled_penalty"])
        next_threshold = None
        if self.stage_index + 1 < len(self.STAGES):
            next_threshold = self.STAGES[self.stage_index + 1]["threshold_m"]
        print(
            "penalty_curriculum "
            f"reason={reason} "
            f"stage={self.stage_index} "
            f"off_track_penalty={env_config.off_track_penalty:.1f} "
            f"stalled_penalty={env_config.stalled_penalty:.1f} "
            f"window={len(self.recent_progress_m)}/{self.window_episodes} "
            f"required={self.required_episodes} "
            f"next_threshold_m={next_threshold}",
            flush=True,
        )

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"penalty_curriculum_state_load_failed path={self.state_path} error={exc}", flush=True)
            return
        try:
            stage_index = int(data.get("stage_index", 0))
        except (TypeError, ValueError):
            stage_index = 0
        self.stage_index = max(0, min(stage_index, len(self.STAGES) - 1))
        progress_values = data.get("recent_progress_m", [])
        if isinstance(progress_values, list):
            self.recent_progress_m.clear()
            for value in progress_values[-self.window_episodes :]:
                try:
                    self.recent_progress_m.append(float(value))
                except (TypeError, ValueError):
                    continue

    def _save_state(self) -> None:
        data = {
            "stage_index": self.stage_index,
            "window_episodes": self.window_episodes,
            "required_episodes": self.required_episodes,
            "recent_progress_m": list(self.recent_progress_m),
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"penalty_curriculum_state_save_failed path={self.state_path} error={exc}", flush=True)


class ActionStatsCallback(BaseCallback if BaseCallback is not None else object):
    def __init__(self, log_freq: int = 0) -> None:
        if BaseCallback is None:  # pragma: no cover - runtime guard
            raise ModuleNotFoundError("stable-baselines3 callbacks are unavailable")
        super().__init__()
        self.log_freq = max(0, int(log_freq))
        self._raw_mean_batches: list[np.ndarray] = []
        self._raw_std_batches: list[np.ndarray] = []
        self._action_batches: list[np.ndarray] = []
        self._env_action_batches: list[np.ndarray] = []

    def _on_step(self) -> bool:
        if self.log_freq <= 0:
            return True
        actions = self.locals.get("actions")
        if actions is None:
            return True
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        clipped = self.locals.get("clipped_actions")
        if clipped is None:
            clipped_array = np.clip(action_array, -1.0, 1.0)
        else:
            clipped_array = np.asarray(clipped, dtype=np.float32)
            if clipped_array.ndim == 1:
                clipped_array = clipped_array.reshape(1, -1)
        raw_mean, raw_std = self._raw_action_params()
        if raw_mean is not None and raw_std is not None:
            self._raw_mean_batches.append(raw_mean)
            self._raw_std_batches.append(raw_std)
        self._action_batches.append(action_array.copy())
        self._env_action_batches.append(clipped_array.copy())
        if self.num_timesteps > 0 and self.num_timesteps % self.log_freq == 0:
            self._print_and_clear()
        return True

    def _raw_action_params(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if th is None:
            return None, None
        obs_tensor = self.locals.get("obs_tensor")
        if obs_tensor is None or self.model is None:
            return None, None
        policy = self.model.policy
        with th.no_grad():
            if hasattr(policy, "raw_action_params"):
                raw_mean, raw_std = policy.raw_action_params(obs_tensor)
            else:
                features = policy.extract_features(obs_tensor)
                if policy.share_features_extractor:
                    latent_pi, _ = policy.mlp_extractor(features)
                else:
                    pi_features, _ = features
                    latent_pi = policy.mlp_extractor.forward_actor(pi_features)
                raw_mean = policy.action_net(latent_pi)
                log_std = getattr(policy, "log_std", None)
                if log_std is None:
                    return None, None
                raw_std = th.ones_like(raw_mean) * log_std.exp()
        return raw_mean.detach().cpu().numpy(), raw_std.detach().cpu().numpy()

    def _print_and_clear(self) -> None:
        if not self._action_batches:
            return
        actions = np.concatenate(self._action_batches, axis=0)
        env_actions = np.concatenate(self._env_action_batches, axis=0)
        raw_mean = np.concatenate(self._raw_mean_batches, axis=0) if self._raw_mean_batches else actions
        raw_std = np.concatenate(self._raw_std_batches, axis=0) if self._raw_std_batches else np.zeros_like(actions)
        self._raw_mean_batches.clear()
        self._raw_std_batches.clear()
        self._action_batches.clear()
        self._env_action_batches.clear()
        if actions.shape[1] >= 2:
            throttle_brake_index = 0
            steer_index = 1
        else:
            return
        steer_action = env_actions[:, steer_index]
        steer_clip_ratio = float(np.mean(np.abs(actions[:, steer_index] - steer_action) > 1e-6))
        steer_saturation_ratio = float(np.mean(np.abs(steer_action) > 0.999))
        print(
            "action_stats "
            f"total_timesteps={self.num_timesteps} "
            f"raw_steer_mean={float(np.mean(raw_mean[:, steer_index])):+.3f} "
            f"raw_steer_std={float(np.mean(raw_std[:, steer_index])):.3f} "
            f"squashed_steer_mean={float(np.mean(steer_action)):+.3f} "
            f"squashed_steer_min={float(np.min(steer_action)):+.3f} "
            f"squashed_steer_max={float(np.max(steer_action)):+.3f} "
            f"steering_saturation_ratio={steer_saturation_ratio:.3f} "
            f"clip_ratio={steer_clip_ratio:.3f}",
            flush=True,
        )
        throttle_brake_action = env_actions[:, throttle_brake_index]
        throttle_brake_clip_ratio = float(
            np.mean(np.abs(actions[:, throttle_brake_index] - throttle_brake_action) > 1e-6)
        )
        throttle_brake_saturation_ratio = float(np.mean(np.abs(throttle_brake_action) > 0.999))
        print(
            f"raw_throttle_brake_mean={float(np.mean(raw_mean[:, throttle_brake_index])):+.3f} "
            f"raw_throttle_brake_std={float(np.mean(raw_std[:, throttle_brake_index])):.3f} "
            f"squashed_throttle_brake_mean={float(np.mean(throttle_brake_action)):+.3f} "
            f"squashed_throttle_brake_min={float(np.min(throttle_brake_action)):+.3f} "
            f"squashed_throttle_brake_max={float(np.max(throttle_brake_action)):+.3f} "
            f"throttle_brake_saturation_ratio={throttle_brake_saturation_ratio:.3f} "
            f"throttle_brake_clip_ratio={throttle_brake_clip_ratio:.3f}",
            flush=True,
        )


class InjectRuntimeErrorCallback(BaseCallback if BaseCallback is not None else object):
    def __init__(self, trigger_timestep: int) -> None:
        if BaseCallback is None:  # pragma: no cover - runtime guard
            raise ModuleNotFoundError("stable-baselines3 callbacks are unavailable")
        super().__init__()
        self.trigger_timestep = int(trigger_timestep)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.trigger_timestep:
            raise RuntimeError(
                "injected runtime recovery test "
                f"at total_timesteps={self.num_timesteps}"
            )
        return True


class TrainingBeVViewerCallback(BaseCallback if BaseCallback is not None else object):
    def __init__(self, env: GymMoraiEnv, scale: int = 6, fps: int = 15) -> None:
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
            print("BeV viewer disabled: latest observation is not BeV/hybrid.", flush=True)
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
        line_2 = f"scenario={info.get('scenario_name')} reason={reason} mode={self.env_ref.observation_mode}"
        line_3 = "overlay: corridor/boundary/reference/ego   close window to hide viewer"
        y0 = bev.shape[1] * self.scale
        self._screen.blit(self._font.render(line_1, True, (240, 240, 240)), (12, y0 + 10))
        self._screen.blit(self._small_font.render(line_2, True, (190, 190, 190)), (12, y0 + 38))
        self._screen.blit(self._small_font.render(line_3, True, (160, 160, 160)), (12, y0 + 60))
        pygame.display.flip()
        return True

    def _on_training_end(self) -> None:
        if not self._closed and pygame is not None:
            pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO policy on the MORAI RL environment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--save-dir", default="runs/ppo_morai")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy", default="auto")
    parser.add_argument("--features-extractor", choices=["auto", "bev_light", "roach", "default"], default="auto")
    parser.add_argument("--action-dist", choices=["gaussian", "tanh_squashed"], default="gaussian")
    parser.add_argument("--std-init", type=float, default=0.1)
    parser.add_argument("--log-std-init", type=float, default=None)
    parser.add_argument("--set-log-std", type=float, default=None)
    parser.add_argument("--set-throttle-brake-mean", type=float, default=None)
    parser.add_argument("--set-throttle-brake-std", type=float, default=None)
    parser.add_argument("--set-steering-std", type=float, default=None)
    parser.add_argument("--action-log-freq", type=int, default=0)
    parser.add_argument("--initial-step-log-count", type=int, default=0)
    parser.add_argument(
        "--stats-dir",
        default="",
        help="Directory for episode CSV/JSONL logs. Defaults to <save-dir>/<run-name>/stats.",
    )
    parser.add_argument("--disable-episode-stats", action="store_true")
    parser.add_argument(
        "--scenario-stats-every",
        type=int,
        default=10,
        help="Print aggregate per-scenario episode stats every N completed episodes. 0 disables.",
    )
    parser.add_argument(
        "--disable-penalty-curriculum",
        action="store_true",
        help="Disable automatic off-track/stalled penalty curriculum updates.",
    )
    parser.add_argument(
        "--penalty-curriculum-window",
        type=int,
        default=39,
        help="Number of recent train episodes used for penalty curriculum stage checks.",
    )
    parser.add_argument(
        "--penalty-curriculum-required",
        type=int,
        default=34,
        help="Required episodes in the curriculum window that must exceed the next progress threshold.",
    )
    parser.add_argument("--sb3-verbose", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--show-bev", action="store_true")
    parser.add_argument("--bev-scale", type=int, default=6)
    parser.add_argument("--bev-fps", type=int, default=15)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--max-restarts", type=int, default=100)
    parser.add_argument("--restart-wait-sec", type=float, default=2.0)
    parser.add_argument("--runtime-recovery", action="store_true")
    parser.add_argument("--disable-runtime-recovery", action="store_true")
    parser.add_argument("--simulator-process-name", action="append", default=[])
    parser.add_argument("--simulator-terminate-timeout-sec", type=float, default=None)
    parser.add_argument("--simulator-relaunch-command", default="")
    parser.add_argument("--simulator-relaunch-wait-sec", type=float, default=None)
    parser.add_argument(
        "--inject-runtime-error-after-steps",
        type=int,
        default=0,
        help="Testing only: raise one RuntimeError at this absolute total timestep.",
    )
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


def _resolve_policy(args: argparse.Namespace, env):
    policy_name = _resolve_policy_name(args, env)
    if args.action_dist != "tanh_squashed":
        return policy_name, policy_name, policy_name
    if policy_name == "MlpPolicy":
        return SquashedActorCriticPolicy, "SquashedMlpPolicy", policy_name
    if policy_name == "CnnPolicy":
        return SquashedActorCriticCnnPolicy, "SquashedCnnPolicy", policy_name
    if policy_name == "MultiInputPolicy":
        return SquashedMultiInputActorCriticPolicy, "SquashedMultiInputPolicy", policy_name
    raise ValueError(
        f"--action-dist tanh_squashed only supports MlpPolicy, CnnPolicy, or MultiInputPolicy; got {policy_name!r}"
    )


def _should_use_bev_light(args: argparse.Namespace, env, policy_name: str) -> bool:
    if args.features_extractor in {"bev_light", "roach"}:
        return True
    if args.features_extractor == "default":
        return False
    return policy_name == "MultiInputPolicy" and isinstance(env.observation_space, gym.spaces.Dict)


def _build_policy_kwargs(args: argparse.Namespace, env, policy_name: str) -> dict:
    policy_kwargs = {}
    if _should_use_bev_light(args, env, policy_name):
        policy_kwargs["features_extractor_class"] = BeVLightCombinedExtractor
    log_std_init = args.log_std_init
    if log_std_init is None and args.std_init is not None and args.std_init > 0.0:
        log_std_init = math.log(float(args.std_init))
    if log_std_init is not None:
        policy_kwargs["log_std_init"] = float(log_std_init)
    return policy_kwargs


def _build_model(args: argparse.Namespace, env, save_dir: Path):
    policy, policy_name, base_policy_name = _resolve_policy(args, env)
    print(f"policy={policy_name}", flush=True)
    print(f"action_dist={args.action_dist}", flush=True)
    policy_kwargs = _build_policy_kwargs(args, env, base_policy_name)
    if "features_extractor_class" in policy_kwargs:
        print("features_extractor=bev_light", flush=True)
    else:
        print("features_extractor=default", flush=True)
    if "log_std_init" in policy_kwargs:
        log_std_init = float(policy_kwargs["log_std_init"])
        print(f"log_std_init={log_std_init:+.4f} std_init={math.exp(log_std_init):.4f}", flush=True)

    if args.resume_from:
        resume_path = _resolve_resume_path(args.resume_from)
        print(f"resuming_from={resume_path}", flush=True)
        if policy_kwargs:
            print("policy_kwargs ignored when resuming from a saved model", flush=True)
        model = PPO.load(str(resume_path), env=env, device=args.device)
        model.verbose = args.sb3_verbose
        model.tensorboard_log = str(save_dir / "tb")
        if args.action_dist == "tanh_squashed":
            _enable_squashed_action_dist(model)
        return model

    return PPO(
        policy=policy,
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=args.sb3_verbose,
        tensorboard_log=str(save_dir / "tb"),
        device=args.device,
        policy_kwargs=policy_kwargs,
    )


def _enable_squashed_action_dist(model) -> None:
    if th is None:
        return
    action_space = model.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError("tanh-squashed actions require a continuous Box action space")
    if not (np.allclose(action_space.low, -1.0) and np.allclose(action_space.high, 1.0)):
        raise ValueError("tanh-squashed PPO policy currently expects Box(-1, 1) actions")
    from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
    from stable_baselines3.common.preprocessing import get_action_dim

    model.policy.action_dist = SquashedDiagGaussianDistribution(get_action_dim(action_space))
    print("converted_loaded_policy_action_dist=tanh_squashed", flush=True)


def _remaining_timesteps(target_timesteps: int, completed_timesteps: int) -> int:
    return max(0, int(target_timesteps) - max(0, int(completed_timesteps)))


def _action_net_row(model, index: int):
    policy = model.policy
    action_net = getattr(policy, "action_net", None)
    if action_net is None or not hasattr(action_net, "weight") or not hasattr(action_net, "bias"):
        raise RuntimeError("policy does not expose an action_net with weight/bias")
    if index < 0 or index >= int(action_net.bias.shape[0]):
        raise IndexError(f"action index {index} out of range for action dim {int(action_net.bias.shape[0])}")
    return action_net, index


def _set_throttle_brake_mean(model, mean: float) -> None:
    if th is None:
        return
    action_net, index = _action_net_row(model, 0)
    with th.no_grad():
        before_bias = float(action_net.bias[index].detach().cpu().item())
        before_weight_norm = float(th.linalg.vector_norm(action_net.weight[index]).detach().cpu().item())
        action_net.weight[index].zero_()
        action_net.bias[index].fill_(float(mean))
        after_bias = float(action_net.bias[index].detach().cpu().item())
        after_weight_norm = float(th.linalg.vector_norm(action_net.weight[index]).detach().cpu().item())
    print(
        "set_throttle_brake_mean "
        f"before_bias={before_bias:+.3f} before_weight_norm={before_weight_norm:.3f} "
        f"after_bias={after_bias:+.3f} after_weight_norm={after_weight_norm:.3f}",
        flush=True,
    )


def _set_action_std(model, index: int, std: float, label: str) -> None:
    if th is None:
        return
    if std <= 0.0:
        raise ValueError(f"{label} std must be positive")
    log_std = getattr(model.policy, "log_std", None)
    if log_std is None:
        raise RuntimeError("policy does not expose log_std")
    if index < 0 or index >= int(log_std.shape[-1]):
        raise IndexError(f"action index {index} out of range for log_std shape {tuple(log_std.shape)}")
    with th.no_grad():
        before_log_std = float(log_std[index].detach().cpu().item())
        before_std = math.exp(before_log_std)
        log_std[index].fill_(math.log(float(std)))
        after_log_std = float(log_std[index].detach().cpu().item())
        after_std = math.exp(after_log_std)
    print(
        f"set_{label}_std index={index} "
        f"before_log_std={before_log_std:+.3f} before_std={before_std:.3f} "
        f"after_log_std={after_log_std:+.3f} after_std={after_std:.3f}",
        flush=True,
    )


def _set_log_std(model, log_std_value: float) -> None:
    if th is None:
        return
    log_std = getattr(model.policy, "log_std", None)
    if log_std is None:
        raise RuntimeError("policy does not expose log_std")
    with th.no_grad():
        before_mean = float(log_std.detach().cpu().mean().item())
        before_std_mean = math.exp(before_mean)
        log_std.fill_(float(log_std_value))
        after_mean = float(log_std.detach().cpu().mean().item())
        after_std_mean = math.exp(after_mean)
    print(
        "set_log_std "
        f"before_mean={before_mean:+.3f} before_std_mean={before_std_mean:.3f} "
        f"after_mean={after_mean:+.3f} after_std_mean={after_std_mean:.3f}",
        flush=True,
    )


def _apply_policy_overrides(model, args: argparse.Namespace) -> None:
    if args.set_log_std is not None:
        _set_log_std(model, float(args.set_log_std))
    if args.set_throttle_brake_mean is not None:
        _set_throttle_brake_mean(model, float(args.set_throttle_brake_mean))
    if args.set_throttle_brake_std is not None:
        _set_action_std(model, 0, float(args.set_throttle_brake_std), "throttle_brake")
    if args.set_steering_std is not None:
        _set_action_std(model, 1, float(args.set_steering_std), "steering")


def _runtime_recovery_options(args: argparse.Namespace) -> dict:
    app_config = load_config(args.config)
    config = app_config.recovery
    enabled = bool(config.enabled)
    if args.runtime_recovery:
        enabled = True
    if args.disable_runtime_recovery:
        enabled = False
    process_names = list(args.simulator_process_name or config.simulator_process_names)
    cleanup_cmdline_substrings = list(config.relaunch_cleanup_cmdline_substrings)
    terminate_timeout_sec = (
        float(config.terminate_timeout_sec)
        if args.simulator_terminate_timeout_sec is None
        else float(args.simulator_terminate_timeout_sec)
    )
    relaunch_command = args.simulator_relaunch_command or config.relaunch_command
    relaunch_wait_sec = (
        float(config.relaunch_wait_sec)
        if args.simulator_relaunch_wait_sec is None
        else float(args.simulator_relaunch_wait_sec)
    )
    required_services = [
        app_config.ros.sync_mode_cmd_service,
        app_config.ros.sync_ctrl_cmd_service,
        app_config.ros.wait_for_tick_service,
        app_config.ros.sync_set_gear_service,
        app_config.ros.sync_scenario_load_service,
    ]
    return {
        "enabled": enabled,
        "process_names": process_names,
        "cleanup_cmdline_substrings": cleanup_cmdline_substrings,
        "terminate_timeout_sec": terminate_timeout_sec,
        "relaunch_command": relaunch_command,
        "relaunch_wait_sec": relaunch_wait_sec,
        "wait_for_services": bool(config.wait_for_services),
        "wait_services_timeout_sec": float(config.wait_services_timeout_sec),
        "wait_services_poll_sec": float(config.wait_services_poll_sec),
        "post_services_wait_sec": float(config.post_services_wait_sec),
        "required_services": required_services,
    }


def _recover_runtime_after_crash(options: dict) -> None:
    if not options.get("enabled", False):
        return
    process_names = [str(name).strip() for name in options.get("process_names", []) if str(name).strip()]
    cleanup_cmdline_substrings = [
        str(pattern).strip()
        for pattern in options.get("cleanup_cmdline_substrings", [])
        if str(pattern).strip()
    ]
    terminate_timeout_sec = float(options.get("terminate_timeout_sec", 5.0))
    relaunch_command = str(options.get("relaunch_command", "")).strip()
    relaunch_wait_sec = max(0.0, float(options.get("relaunch_wait_sec", 0.0)))
    print(
        "runtime_recovery start "
        f"process_names={process_names} "
        f"cleanup_cmdline_substrings={cleanup_cmdline_substrings} "
        f"relaunch_command={relaunch_command!r}",
        flush=True,
    )
    killed_pids = terminate_processes_by_name(process_names, terminate_timeout_sec)
    print(f"runtime_recovery killed_pids={killed_pids}", flush=True)
    cleanup_pids = terminate_processes_by_cmdline_substrings(
        cleanup_cmdline_substrings,
        terminate_timeout_sec,
    )
    print(f"runtime_recovery cleanup_pids={cleanup_pids}", flush=True)
    launched_pid = launch_process(relaunch_command) if relaunch_command else None
    print(f"runtime_recovery launched_pid={launched_pid}", flush=True)
    if relaunch_wait_sec > 0.0:
        print(f"runtime_recovery waiting {relaunch_wait_sec:.1f}s", flush=True)
        time.sleep(relaunch_wait_sec)
    if options.get("wait_for_services", True):
        _wait_for_ros_services(
            [str(service) for service in options.get("required_services", [])],
            timeout_sec=float(options.get("wait_services_timeout_sec", 90.0)),
            poll_sec=float(options.get("wait_services_poll_sec", 2.0)),
        )
    post_services_wait_sec = max(0.0, float(options.get("post_services_wait_sec", 0.0)))
    if post_services_wait_sec > 0.0:
        print(f"runtime_recovery post_services_wait {post_services_wait_sec:.1f}s", flush=True)
        time.sleep(post_services_wait_sec)


def _wait_for_ros_services(required_services: list[str], timeout_sec: float, poll_sec: float) -> None:
    required = {service.strip() for service in required_services if service.strip()}
    if not required:
        return
    deadline = time.monotonic() + max(0.0, timeout_sec)
    last_missing = sorted(required)
    while time.monotonic() <= deadline:
        try:
            result = subprocess.run(
                ["rosservice", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1.0, min(10.0, poll_sec)),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_missing = sorted(required)
            print(f"runtime_recovery rosservice_list_failed error={exc}", flush=True)
        else:
            services = {line.strip() for line in result.stdout.splitlines() if line.strip()}
            missing = sorted(required - services)
            if not missing:
                print(
                    "runtime_recovery services_ready "
                    f"services={sorted(required)}",
                    flush=True,
                )
                return
            last_missing = missing
            print(f"runtime_recovery waiting_for_services missing={missing}", flush=True)
        time.sleep(max(0.1, poll_sec))
    raise RuntimeError(f"runtime recovery timed out waiting for ROS services: {last_missing}")


def _close_env_quietly(env) -> None:
    try:
        env.close()
    except Exception as exc:
        print(f"env_close_failed_during_restart error={exc}", flush=True)


def main() -> None:
    if PPO is None or Monitor is None or CheckpointCallback is None or CallbackList is None:
        raise ModuleNotFoundError(
            "stable-baselines3 and torch are required. Install them with "
            "`pip install stable-baselines3 gymnasium torch`."
        ) from _SB3_IMPORT_ERROR

    args = parse_args()
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = None if args.disable_episode_stats else Path(args.stats_dir or (save_dir / "stats"))
    if stats_dir is not None:
        stats_dir.mkdir(parents=True, exist_ok=True)
        print(f"episode_stats_dir={stats_dir}", flush=True)
    recovery_options = _runtime_recovery_options(args)
    if recovery_options["enabled"]:
        print(
            "runtime_recovery enabled "
            f"process_names={recovery_options['process_names']} "
            f"relaunch_command={recovery_options['relaunch_command']!r}",
            flush=True,
        )
    base_env = GymMoraiEnv(args.config)
    env = Monitor(base_env)
    model = _build_model(args, env, save_dir)
    _apply_policy_overrides(model, args)
    restart_count = 0
    current_resume_from = args.resume_from
    reset_num_timesteps = not bool(current_resume_from)
    injected_runtime_error = False

    while True:
        remaining_timesteps = _remaining_timesteps(args.timesteps, model.num_timesteps)
        if remaining_timesteps <= 0:
            print(
                "target_timesteps_already_reached "
                f"target={args.timesteps} completed={model.num_timesteps}",
                flush=True,
            )
            model.save(str(save_dir / "ppo_model"))
            print(f"saved_model={save_dir / 'ppo_model'}", flush=True)
            break

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_checkpoint",
        )
        episode_stats_callback = EpisodeResetStatsCallback(
            initial_step_log_count=args.initial_step_log_count,
            stats_dir=stats_dir,
            scenario_stats_every=args.scenario_stats_every,
        )
        callbacks = [checkpoint_callback, episode_stats_callback]
        if not args.disable_penalty_curriculum:
            callbacks.append(
                PenaltyCurriculumCallback(
                    env=base_env,
                    state_path=save_dir / "penalty_curriculum_state.json",
                    window_episodes=args.penalty_curriculum_window,
                    required_episodes=args.penalty_curriculum_required,
                )
            )
        if args.action_log_freq > 0:
            callbacks.append(ActionStatsCallback(log_freq=args.action_log_freq))
        if args.show_bev:
            callbacks.append(TrainingBeVViewerCallback(env=base_env, scale=args.bev_scale, fps=args.bev_fps))
        if (
            args.inject_runtime_error_after_steps > 0
            and not injected_runtime_error
            and model.num_timesteps < args.inject_runtime_error_after_steps <= args.timesteps
        ):
            callbacks.append(InjectRuntimeErrorCallback(args.inject_runtime_error_after_steps))
        callback = CallbackList(callbacks)
        try:
            print(
                "training_budget "
                f"target={args.timesteps} completed={model.num_timesteps} "
                f"remaining={remaining_timesteps}",
                flush=True,
            )
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callback,
                progress_bar=args.progress_bar,
                reset_num_timesteps=reset_num_timesteps,
            )
            model.save(str(save_dir / "ppo_model"))
            print(f"saved_model={save_dir / 'ppo_model'}", flush=True)
            break
        except KeyboardInterrupt:
            model.save(str(save_dir / "ppo_model_interrupted"))
            print(f"saved_model={save_dir / 'ppo_model_interrupted'}", flush=True)
            raise
        except RuntimeError as exc:
            if str(exc).startswith("injected runtime recovery test"):
                injected_runtime_error = True
            model.save(str(save_dir / "ppo_model_crash"))
            print(f"saved_model={save_dir / 'ppo_model_crash'}", flush=True)
            restart_count += 1
            if restart_count > args.max_restarts:
                raise RuntimeError(f"training aborted after {restart_count} restarts") from exc
            completed_timesteps = model.num_timesteps
            remaining_timesteps = _remaining_timesteps(args.timesteps, completed_timesteps)
            print(
                "training crashed, restarting from latest crash model "
                f"({restart_count}/{args.max_restarts}, "
                f"num_timesteps={completed_timesteps}, "
                f"remaining_timesteps={remaining_timesteps}, error={exc})",
                flush=True,
            )
            _close_env_quietly(env)
            if recovery_options["enabled"]:
                _recover_runtime_after_crash(recovery_options)
            elif args.restart_wait_sec > 0.0:
                time.sleep(args.restart_wait_sec)
            current_resume_from = str(save_dir / "ppo_model_crash.zip")
            args.resume_from = current_resume_from
            base_env = GymMoraiEnv(args.config)
            env = Monitor(base_env)
            model = _build_model(args, env, save_dir)
            _apply_policy_overrides(model, args)
            reset_num_timesteps = False
        except Exception:
            model.save(str(save_dir / "ppo_model_crash"))
            print(f"saved_model={save_dir / 'ppo_model_crash'}", flush=True)
            raise

    _close_env_quietly(env)


if __name__ == "__main__":
    main()
