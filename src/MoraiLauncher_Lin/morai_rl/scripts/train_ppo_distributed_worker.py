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
    parser.add_argument("--std-init", type=float, default=0.1)
    parser.add_argument("--log-std-init", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    if th is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("torch is required") from _TORCH_IMPORT_ERROR
    args = parse_args()
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
        std_init=args.std_init,
        log_std_init=args.log_std_init,
    )

    sock = _connect(args.server_host, args.server_port, args.connect_retry_sec)
    send_message(sock, {"type": "hello", "worker_id": args.worker_id})
    obs, _info = env.reset()
    episode_start = True

    try:
        while True:
            message = recv_message(sock)
            if message.get("type") != "policy":
                raise RuntimeError(f"expected policy message, got {message.get('type')!r}")
            policy_version = int(message["policy_version"])
            load_policy_state(model, message["policy_state"])
            print(
                "policy_loaded "
                f"worker_id={args.worker_id} policy_version={policy_version} "
                f"checksum={str(message.get('policy_checksum', ''))[:12]}",
                flush=True,
            )
            rollout, obs, episode_start = _collect_rollout(
                env=env,
                model=model,
                initial_obs=obs,
                initial_episode_start=episode_start,
                rollout_steps=args.rollout_steps,
            )
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


def _collect_rollout(
    *,
    env: GymMoraiEnv,
    model,
    initial_obs,
    initial_episode_start: bool,
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

    for _ in range(int(rollout_steps)):
        action, value, log_prob = _sample_action(model, obs)
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, info = env.step(clipped_action)
        done = bool(terminated or truncated)

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
        },
        obs,
        episode_start,
    )


def _sample_action(model, obs):
    with th.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        actions_tensor, values_tensor, log_probs_tensor = model.policy(obs_tensor)
    action = actions_tensor.detach().cpu().numpy().reshape(-1)
    value = values_tensor.detach().cpu().numpy().reshape(-1)[0]
    log_prob = log_probs_tensor.detach().cpu().numpy().reshape(-1)[0]
    return action, value, log_prob


def _predict_value(model, obs) -> float:
    with th.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        value = model.policy.predict_values(obs_tensor)
    return float(value.detach().cpu().numpy().reshape(-1)[0])


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


if __name__ == "__main__":
    main()
