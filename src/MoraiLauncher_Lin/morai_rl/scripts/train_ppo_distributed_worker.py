from __future__ import annotations

import argparse
from pathlib import Path
import socket
import time

import numpy as np

try:
    import torch as th
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    th = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from morai_rl.distributed.model import build_distributed_ppo, load_policy_state
from morai_rl.distributed.protocol import recv_message, send_message
from morai_rl.envs.gym_wrapper import GymMoraiEnv
from morai_rl.scripts.train_ppo import (
    _close_env_quietly,
    _recover_runtime_after_crash,
    _runtime_recovery_options,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "stage1_ros_sync_config.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronous distributed PPO worker for MORAI RL.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--server-port", type=int, default=50051)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--connect-retry-sec", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy", default="auto")
    parser.add_argument("--features-extractor", choices=["auto", "roach", "default"], default="auto")
    parser.add_argument("--action-dist", choices=["gaussian", "tanh_squashed"], default="gaussian")
    parser.add_argument("--std-init", type=float, default=0.1)
    parser.add_argument("--log-std-init", type=float, default=None)
    parser.add_argument("--action-log-freq", type=int, default=0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--restart-wait-sec", type=float, default=2.0)
    parser.add_argument("--runtime-recovery", action="store_true")
    parser.add_argument("--disable-runtime-recovery", action="store_true")
    parser.add_argument("--simulator-process-name", action="append", default=[])
    parser.add_argument("--simulator-terminate-timeout-sec", type=float, default=None)
    parser.add_argument("--simulator-relaunch-command", default="")
    parser.add_argument("--simulator-relaunch-wait-sec", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    if th is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("torch is required") from _TORCH_IMPORT_ERROR
    args = parse_args()
    recovery_options = _runtime_recovery_options(args)
    if recovery_options["enabled"]:
        print(
            "worker_runtime_recovery enabled "
            f"worker_id={args.worker_id} process_names={recovery_options['process_names']} "
            f"relaunch_command={recovery_options['relaunch_command']!r}",
            flush=True,
        )
    env = GymMoraiEnv(args.config)
    model = build_distributed_ppo(
        config_path=args.config,
        n_envs=1,
        n_steps=args.rollout_steps,
        batch_size=min(512, args.rollout_steps),
        n_epochs=1,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        device=args.device,
        policy=args.policy,
        features_extractor=args.features_extractor,
        action_dist=args.action_dist,
        std_init=args.std_init,
        log_std_init=args.log_std_init,
    )

    sock = _connect(args.server_host, args.server_port, args.connect_retry_sec)
    send_message(sock, {"type": "hello", "worker_id": args.worker_id})
    obs, _info = env.reset()
    episode_start = True
    episode_tracker = _new_episode_tracker()
    action_stats = _ActionStats(args.worker_id, log_freq=args.action_log_freq)

    try:
        while True:
            try:
                message = recv_message(sock)
            except EOFError as exc:
                print(f"learner_disconnected worker_id={args.worker_id} error={exc}", flush=True)
                break
            if message.get("type") == "shutdown":
                print(f"shutdown_received worker_id={args.worker_id}", flush=True)
                break
            if message.get("type") != "policy":
                raise RuntimeError(f"expected policy message, got {message.get('type')!r}")
            policy_version = int(message["policy_version"])
            load_policy_state(model, message["policy_state"])
            penalty_curriculum = message.get("penalty_curriculum")
            _apply_penalty_curriculum(env, penalty_curriculum, args.worker_id)
            print(
                "policy_loaded "
                f"worker_id={args.worker_id} policy_version={policy_version} "
                f"checksum={str(message.get('policy_checksum', ''))[:12]}",
                flush=True,
            )
            restart_count = 0
            while True:
                try:
                    rollout, obs, episode_start = _collect_rollout(
                        env=env,
                        model=model,
                        worker_id=args.worker_id,
                        initial_obs=obs,
                        initial_episode_start=episode_start,
                        episode_tracker=episode_tracker,
                        action_stats=action_stats,
                        rollout_steps=args.rollout_steps,
                    )
                    break
                except RuntimeError as exc:
                    restart_count += 1
                    if restart_count > args.max_restarts:
                        _send_worker_error(sock, args.worker_id, policy_version, exc)
                        raise RuntimeError(
                            f"worker {args.worker_id} aborted policy_version={policy_version} "
                            f"after {restart_count} rollout restarts"
                        ) from exc
                    print(
                        "worker_rollout_runtime_error "
                        f"worker_id={args.worker_id} policy_version={policy_version} "
                        f"restart={restart_count}/{args.max_restarts} error={exc}",
                        flush=True,
                    )
                    _close_env_quietly(env)
                    if recovery_options["enabled"]:
                        _recover_runtime_after_crash(recovery_options)
                    elif args.restart_wait_sec > 0.0:
                        time.sleep(args.restart_wait_sec)
                    env = GymMoraiEnv(args.config)
                    _apply_penalty_curriculum(env, penalty_curriculum, args.worker_id)
                    obs, _info = env.reset()
                    episode_start = True
                    episode_tracker = _new_episode_tracker()
                    action_stats = _ActionStats(args.worker_id, log_freq=args.action_log_freq)
            send_message(
                sock,
                {
                    "type": "rollout",
                    "worker_id": args.worker_id,
                    "policy_version": policy_version,
                    "rollout": rollout,
                },
            )
            print(
                f"rollout_sent worker_id={args.worker_id} policy_version={policy_version} "
                f"steps={rollout['steps']}",
                flush=True,
            )
    finally:
        env.close()
        sock.close()


def _connect(host: str, port: int, retry_sec: float) -> socket.socket:
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=10.0)
        except OSError as exc:
            print(f"connect_failed host={host} port={port} error={exc}", flush=True)
            time.sleep(max(0.1, retry_sec))
            continue
        sock.settimeout(None)
        print(f"connected_to_learner host={host} port={port}", flush=True)
        return sock


def _apply_penalty_curriculum(env: GymMoraiEnv, payload, worker_id: str) -> None:
    if not isinstance(payload, dict):
        return
    try:
        off_track_penalty = float(payload["off_track_penalty"])
        stalled_penalty = float(payload["stalled_penalty"])
    except (KeyError, TypeError, ValueError):
        print(
            f"worker_penalty_curriculum_invalid worker_id={worker_id} payload={payload}",
            flush=True,
        )
        return
    env_config = env.env.config.env
    env_config.off_track_penalty = off_track_penalty
    env_config.stalled_penalty = stalled_penalty
    print(
        "worker_penalty_curriculum_applied "
        f"worker_id={worker_id} "
        f"stage={payload.get('stage_index')} "
        f"off_track_penalty={off_track_penalty:.1f} "
        f"stalled_penalty={stalled_penalty:.1f}",
        flush=True,
    )


def _send_worker_error(sock: socket.socket, worker_id: str, policy_version: int, exc: Exception) -> None:
    try:
        send_message(
            sock,
            {
                "type": "worker_error",
                "worker_id": worker_id,
                "policy_version": int(policy_version),
                "error": repr(exc),
            },
        )
    except OSError as send_exc:
        print(f"worker_error_send_failed worker_id={worker_id} error={send_exc}", flush=True)


def _collect_rollout(
    *,
    env: GymMoraiEnv,
    model,
    worker_id: str,
    initial_obs,
    initial_episode_start: bool,
    episode_tracker: dict,
    action_stats,
    rollout_steps: int,
) -> tuple[dict, object, bool]:
    model.policy.set_training_mode(False)
    obs = initial_obs
    episode_start = bool(initial_episode_start)
    observations = []
    actions = []
    rewards = []
    episode_starts = []
    values = []
    log_probs = []
    terminateds = []
    truncateds = []
    infos = []
    episode_summaries = []

    for _ in range(int(rollout_steps)):
        action, value, log_prob, raw_mean, raw_std = _sample_action(model, obs)
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, info = env.step(clipped_action)
        done = bool(terminated or truncated)
        _update_episode_tracker(episode_tracker, reward, info)
        action_stats.add(raw_mean, raw_std, action, clipped_action)

        observations.append(_copy_observation(obs))
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(reward))
        episode_starts.append(float(episode_start))
        values.append(float(value))
        log_probs.append(float(log_prob))
        terminateds.append(bool(terminated))
        truncateds.append(bool(truncated))
        infos.append(_compact_info(info))

        if done:
            episode_summaries.append(_finish_episode_tracker(episode_tracker, info, worker_id))
            obs, _reset_info = env.reset()
            episode_start = True
        else:
            obs = next_obs
            episode_start = False

    last_value = _predict_value(model, obs)
    return (
        {
            "steps": int(rollout_steps),
            "obs": observations,
            "actions": actions,
            "rewards": rewards,
            "episode_starts": episode_starts,
            "values": values,
            "log_probs": log_probs,
            "terminateds": terminateds,
            "truncateds": truncateds,
            "last_obs": _copy_observation(obs),
            "last_value": float(last_value),
            "last_episode_start": float(episode_start),
            "infos": infos,
            "episode_summaries": episode_summaries,
        },
        obs,
        episode_start,
    )


def _sample_action(model, obs):
    with th.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        raw_mean, raw_std = _raw_action_params(model, obs_tensor)
        actions_tensor, values_tensor, log_probs_tensor = model.policy(obs_tensor)
    action = actions_tensor.detach().cpu().numpy().reshape(-1)
    value = values_tensor.detach().cpu().numpy().reshape(-1)[0]
    log_prob = log_probs_tensor.detach().cpu().numpy().reshape(-1)[0]
    return action, value, log_prob, raw_mean, raw_std


def _raw_action_params(model, obs_tensor):
    policy = model.policy
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
    return raw_mean.detach().cpu().numpy().reshape(-1), raw_std.detach().cpu().numpy().reshape(-1)


def _predict_value(model, obs) -> float:
    with th.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        value = model.policy.predict_values(obs_tensor)
    return float(value.detach().cpu().numpy().reshape(-1)[0])


class _ActionStats:
    def __init__(self, worker_id: str, log_freq: int = 0) -> None:
        self.worker_id = str(worker_id)
        self.log_freq = max(0, int(log_freq))
        self.total_steps = 0
        self._raw_means: list[np.ndarray] = []
        self._raw_stds: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._env_actions: list[np.ndarray] = []

    def add(self, raw_mean, raw_std, action, env_action) -> None:
        if self.log_freq <= 0:
            return
        self.total_steps += 1
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        env_action_array = np.asarray(env_action, dtype=np.float32).reshape(-1)
        self._actions.append(action_array.copy())
        self._env_actions.append(env_action_array.copy())
        if raw_mean is not None and raw_std is not None:
            self._raw_means.append(np.asarray(raw_mean, dtype=np.float32).reshape(-1).copy())
            self._raw_stds.append(np.asarray(raw_std, dtype=np.float32).reshape(-1).copy())
        if self.total_steps % self.log_freq == 0:
            self._print_and_clear()

    def _print_and_clear(self) -> None:
        if not self._actions:
            return
        actions = np.stack(self._actions)
        env_actions = np.stack(self._env_actions)
        raw_mean = np.stack(self._raw_means) if self._raw_means else actions
        raw_std = np.stack(self._raw_stds) if self._raw_stds else np.zeros_like(actions)
        self._raw_means.clear()
        self._raw_stds.clear()
        self._actions.clear()
        self._env_actions.clear()
        if actions.shape[1] < 2:
            return
        throttle_brake_index = 0
        steer_index = 1
        steer_action = env_actions[:, steer_index]
        steer_clip_ratio = float(np.mean(np.abs(actions[:, steer_index] - steer_action) > 1e-6))
        steer_saturation_ratio = float(np.mean(np.abs(steer_action) > 0.999))
        print(
            "worker_action_stats "
            f"worker_id={self.worker_id} total_steps={self.total_steps} "
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
            "worker_action_stats "
            f"worker_id={self.worker_id} total_steps={self.total_steps} "
            f"raw_throttle_brake_mean={float(np.mean(raw_mean[:, throttle_brake_index])):+.3f} "
            f"raw_throttle_brake_std={float(np.mean(raw_std[:, throttle_brake_index])):.3f} "
            f"squashed_throttle_brake_mean={float(np.mean(throttle_brake_action)):+.3f} "
            f"squashed_throttle_brake_min={float(np.min(throttle_brake_action)):+.3f} "
            f"squashed_throttle_brake_max={float(np.max(throttle_brake_action)):+.3f} "
            f"throttle_brake_saturation_ratio={throttle_brake_saturation_ratio:.3f} "
            f"throttle_brake_clip_ratio={throttle_brake_clip_ratio:.3f}",
            flush=True,
        )


def _copy_observation(obs):
    if isinstance(obs, dict):
        return {key: np.asarray(value).copy() for key, value in obs.items()}
    return np.asarray(obs).copy()


def _compact_info(info: dict) -> dict:
    return {
        "scenario_name": info.get("scenario_name"),
        "termination_reason": info.get("termination_reason"),
        "step_count": info.get("step_count"),
        "episode_progress_m": info.get("episode_progress_m"),
        "progress_delta_m": info.get("progress_delta_m"),
        "reward_terms": info.get("reward_terms"),
    }


def _new_episode_tracker() -> dict:
    return {
        "episode_count": 0,
        "total_timesteps": 0,
        "episode_reward": 0.0,
        "reward_terms": {},
    }


def _update_episode_tracker(tracker: dict, reward: float, info: dict) -> None:
    tracker["total_timesteps"] = int(tracker.get("total_timesteps", 0)) + 1
    tracker["episode_reward"] = float(tracker.get("episode_reward", 0.0)) + float(reward)
    reward_terms = info.get("reward_terms")
    if not isinstance(reward_terms, dict):
        return
    term_sums = tracker.setdefault("reward_terms", {})
    for key, value in reward_terms.items():
        if isinstance(value, (int, float)):
            term_sums[key] = float(term_sums.get(key, 0.0)) + float(value)


def _finish_episode_tracker(tracker: dict, info: dict, worker_id: str) -> dict:
    tracker["episode_count"] = int(tracker.get("episode_count", 0)) + 1
    summary = {
        "episode_count": int(tracker.get("episode_count", 0)),
        "total_timesteps": int(tracker.get("total_timesteps", 0)),
        "scenario_name": info.get("scenario_name"),
        "termination_reason": info.get("termination_reason"),
        "step_count": info.get("step_count"),
        "episode_progress_m": info.get("episode_progress_m"),
        "episode_reward": float(tracker.get("episode_reward", 0.0)),
        "episode_duration_sec": info.get("episode_duration_sec"),
        "lap_completed": info.get("lap_completed"),
        "reward_terms": dict(tracker.get("reward_terms", {})),
    }
    _print_episode_summary(worker_id, summary)
    tracker["episode_reward"] = 0.0
    tracker["reward_terms"] = {}
    return summary


def _print_episode_summary(worker_id: str, summary: dict) -> None:
    print(
        "worker_episode_end "
        f"worker_id={worker_id} "
        f"count={summary.get('episode_count')} "
        f"total_timesteps={summary.get('total_timesteps')} "
        f"scenario={summary.get('scenario_name')} "
        f"reason={summary.get('termination_reason')} "
        f"steps={summary.get('step_count')}",
        flush=True,
    )
    _print_float_line("  progress_m", summary.get("episode_progress_m"))
    _print_float_line("  total_reward", summary.get("episode_reward"))
    reward_terms = summary.get("reward_terms")
    if not isinstance(reward_terms, dict) or not reward_terms:
        return
    print("  reward_terms", flush=True)
    for key, value in reward_terms.items():
        if isinstance(value, (int, float)):
            signed_value = float(value)
            if key.endswith("_penalty"):
                signed_value = -signed_value
            print(f"    {key}={signed_value:+.3f}", flush=True)
        else:
            print(f"    {key}={value}", flush=True)


def _print_float_line(label: str, value) -> None:
    try:
        print(f"{label}={float(value):+.3f}", flush=True)
    except (TypeError, ValueError):
        print(f"{label}={value}", flush=True)


if __name__ == "__main__":
    main()
